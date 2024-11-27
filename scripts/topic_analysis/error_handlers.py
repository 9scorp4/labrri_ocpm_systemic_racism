"""Error handling for topic analysis pipeline with async support."""
import psutil
import gc
from typing import Optional, Dict, Any, List, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import traceback
from loguru import logger
import asyncio
from datetime import datetime
import json
from pathlib import Path
import aiofiles
import os
from contextlib import asynccontextmanager

class AnalysisErrorType(Enum):
    """Types of errors that can occur during topic analysis."""
    INITIALIZATION = "initialization"
    DATABASE = "database"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    ASYNC_OPERATION = "async_operation"
    RESOURCE = "resource"
    VALIDATION = "validation"
    LANGUAGE_DETECTION = "language_detection"
    TEXT_PROCESSING = "text_processing"
    TOPIC_MODELING = "topic_modeling"
    CACHE = "cache"
    CLEANUP = "cleanup"
    SERVER = "server"
    MEMORY_ERROR = "memory_error"
    RESOURCE_ERROR = "resource_error"
    VECTOR_LOADING_ERROR = "vector_loading_error"
    MATRIX_ERROR = "matrix_error"
    UNKNOWN = "unknown"

@dataclass
class AnalysisError:
    """Structured error information for topic analysis."""
    error_type: AnalysisErrorType
    message: str
    details: Optional[str] = None
    traceback: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    component: Optional[str] = None
    recoverable: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    error_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            'error_id': self.error_id,
            'type': self.error_type.value,
            'message': self.message,
            'details': self.details,
            'traceback': self.traceback,
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'recoverable': self.recoverable,
            'context': self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisError':
        """Create error from dictionary."""
        return cls(
            error_type=AnalysisErrorType(data['type']),
            message=data['message'],
            details=data.get('details'),
            traceback=data.get('traceback'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            component=data.get('component'),
            recoverable=data.get('recoverable', True),
            context=data.get('context', {}),
            error_id=data.get('error_id')
        )

class ErrorHandler:
    """Handles error management for topic analysis pipeline with async support."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize error handler.
        
        Args:
            log_dir: Directory for error logs
        """
        self.log_dir = log_dir or Path('logs/errors')
        self._error_cache: Dict[str, AnalysisError] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
        
    async def initialize(self):
        """Initialize error handler asynchronously."""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            try:
                # Create log directory synchronously since it's a one-time operation
                os.makedirs(self.log_dir, exist_ok=True)
                
                # Load existing errors
                await self._load_errors()
                
                self._initialized = True
                logger.info("Error handler initialized")
                
            except Exception as e:
                logger.error(f"Error initializing error handler: {e}")
                raise

    async def capture_error(
        self,
        error: Exception,
        error_type: AnalysisErrorType,
        component: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisError:
        """Capture and structure an error asynchronously."""
        await self.initialize()
        
        try:
            # Generate error ID
            error_id = f"{error_type.value}_{datetime.now().timestamp()}"
            
            # Create structured error
            analysis_error = AnalysisError(
                error_type=error_type,
                message=str(error),
                details=str(type(error).__name__),
                traceback=traceback.format_exc(),
                component=component,
                recoverable=error_type not in {
                    AnalysisErrorType.INITIALIZATION,
                    AnalysisErrorType.DATABASE
                },
                context=context or {},
                error_id=error_id
            )
            
            # Cache error
            async with self._lock:
                self._error_cache[error_id] = analysis_error
                
            # Log error
            await self.log_error(analysis_error)
            
            return analysis_error
            
        except Exception as e:
            logger.error(f"Error capturing error: {e}")
            return AnalysisError(
                error_type=AnalysisErrorType.UNKNOWN,
                message=str(error),
                recoverable=False
            )

    async def log_error(self, error: AnalysisError):
        """Log error with appropriate severity asynchronously."""
        try:
            # Format log message
            log_message = self._format_log_message(error)
            
            # Log to file
            log_file = self.log_dir / f"errors_{datetime.now():%Y%m%d}.json"
            async with aiofiles.open(log_file, mode='a') as f:
                await f.write(json.dumps(error.to_dict()) + '\n')
            
            # Log to console with appropriate severity
            if not error.recoverable:
                logger.error(log_message)
                if error.traceback:
                    logger.error(f"Traceback:\n{error.traceback}")
            else:
                logger.warning(log_message)
                
        except Exception as e:
            logger.error(f"Error logging error: {e}")

    def _format_log_message(self, error: AnalysisError) -> str:
        """Format error message for logging."""
        parts = [
            f"Error [{error.error_id}]:",
            f"Type: {error.error_type.value}",
            f"Message: {error.message}"
        ]
        
        if error.component:
            parts.append(f"Component: {error.component}")
            
        if error.details:
            parts.append(f"Details: {error.details}")
            
        if error.context:
            parts.append(f"Context: {json.dumps(error.context, indent=2)}")
            
        return '\n'.join(parts)

    async def get_error(self, error_id: str) -> Optional[AnalysisError]:
        """Get error by ID."""
        async with self._lock:
            return self._error_cache.get(error_id)

    async def list_errors(
        self,
        error_type: Optional[AnalysisErrorType] = None,
        component: Optional[str] = None,
        recoverable: Optional[bool] = None
    ) -> List[AnalysisError]:
        """List errors with optional filtering."""
        async with self._lock:
            errors = list(self._error_cache.values())
            
        if error_type:
            errors = [e for e in errors if e.error_type == error_type]
            
        if component:
            errors = [e for e in errors if e.component == component]
            
        if recoverable is not None:
            errors = [e for e in errors if e.recoverable == recoverable]
            
        return sorted(errors, key=lambda e: e.timestamp, reverse=True)

    async def clear_errors(self, older_than: Optional[datetime] = None):
        """Clear error cache."""
        async with self._lock:
            if older_than:
                self._error_cache = {
                    id: error
                    for id, error in self._error_cache.items()
                    if error.timestamp > older_than
                }
            else:
                self._error_cache.clear()

    async def _load_errors(self):
        """Load existing errors from log files."""
        try:
            # Use regular glob since this is a startup operation
            error_files = list(self.log_dir.glob("errors_*.json"))
            
            for file_path in error_files:
                if not file_path.name.startswith('errors_') or not file_path.name.endswith('.json'):
                    continue
                    
                async with aiofiles.open(file_path) as f:
                    content = await f.read()
                    for line in content.splitlines():
                        try:
                            if line.strip():
                                error_dict = json.loads(line)
                                error = AnalysisError.from_dict(error_dict)
                                self._error_cache[error.error_id] = error
                        except Exception as e:
                            logger.error(f"Error loading error from {file_path}: {e}")
                            
        except Exception as e:
            logger.error(f"Error loading existing errors: {e}")

    @asynccontextmanager
    async def error_context(
        self,
        error_type: AnalysisErrorType,
        component: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            error = await self.capture_error(e, error_type, component, context)
            raise type(e)(str(error.message)) from e

    async def handle_memory_error(
        self,
        error: Exception,
        component: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AnalysisError:
        """Handle memory-related errors."""
        # Force garbage collection
        gc.collect()
        
        error_context = context or {}
        error_context.update({
            'available_memory': psutil.virtual_memory().available,
            'total_memory': psutil.virtual_memory().total,
            'memory_percent': psutil.virtual_memory().percent
        })
        
        analysis_error = AnalysisError(
            error_type=AnalysisErrorType.MEMORY_ERROR,
            message=f"Memory error in {component}: {str(error)}",
            details=str(type(error).__name__),
            component=component,
            context=error_context,
            recoverable=True
        )
        
        await self.log_error(analysis_error)
        return analysis_error

    async def handle_matrix_error(
        self,
        error: Exception,
        shape: Tuple[int, int],
        dtype: np.dtype,
        component: str
    ) -> AnalysisError:
        """Handle matrix operation errors."""
        error_context = {
            'matrix_shape': shape,
            'dtype': str(dtype),
            'required_memory': np.prod(shape) * dtype.itemsize
        }
        
        analysis_error = AnalysisError(
            error_type=AnalysisErrorType.MATRIX_ERROR,
            message=f"Matrix operation error in {component}: {str(error)}",
            details=str(type(error).__name__),
            component=component,
            context=error_context,
            recoverable=True
        )
        
        await self.log_error(analysis_error)
        return analysis_error

    async def cleanup(self):
        """Clean up resources."""
        try:
            await self.clear_errors()
            self._initialized = False
            logger.info("Error handler cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

class ServerErrorHandler:
    """Handles server-level errors for the dashboard."""
    
    def __init__(self, error_handler: ErrorHandler):
        """Initialize server error handler.
        
        Args:
            error_handler: Main error handler instance
        """
        self.error_handler = error_handler
        self.server_errors: List[AnalysisError] = []
        self._lock = asyncio.Lock()
        
    async def handle_server_error(
        self,
        error: Exception,
        context: str = None,
        recoverable: bool = True
    ) -> Optional[AnalysisError]:
        """Handle server-level error.
        
        Args:
            error: Exception that occurred
            context: Error context description
            recoverable: Whether the error is recoverable
            
        Returns:
            Captured analysis error or None if handling failed
        """
        try:
            # Create detailed context
            error_context = {
                "server_context": context,
                "error_time": datetime.now().isoformat(),
                "error_type": error.__class__.__name__,
                "recoverable": recoverable
            }
            
            # Capture error
            analysis_error = await self.error_handler.capture_error(
                error,
                AnalysisErrorType.SERVER,
                component="server",
                context=error_context
            )
            
            async with self._lock:
                self.server_errors.append(analysis_error)
                
            # Log error details
            logger.error(
                f"Server error in {context or 'unknown context'}: "
                f"{error.__class__.__name__}: {str(error)}",
                exc_info=True
            )
            
            # Log recovery guidance if error is recoverable
            if recoverable:
                logger.info(
                    "This error is recoverable. The server will attempt to continue operation."
                )
            else:
                logger.warning(
                    "This error is not recoverable. Server restart may be required."
                )
            
            return analysis_error
            
        except Exception as e:
            logger.error(f"Error in server error handler: {e}", exc_info=True)
            return None
            
    async def get_server_errors(
        self,
        include_resolved: bool = False
    ) -> List[AnalysisError]:
        """Get server errors.
        
        Args:
            include_resolved: Whether to include resolved errors
            
        Returns:
            List of server errors
        """
        async with self._lock:
            if include_resolved:
                return self.server_errors.copy()
            return [
                error for error in self.server_errors
                if not error.context.get("resolved", False)
            ]
            
    async def mark_error_resolved(self, error_id: str):
        """Mark a server error as resolved.
        
        Args:
            error_id: ID of error to mark as resolved
        """
        async with self._lock:
            for error in self.server_errors:
                if error.error_id == error_id:
                    error.context["resolved"] = True
                    error.context["resolved_time"] = datetime.now().isoformat()
                    break
            
    async def clear_server_errors(self, include_unresolved: bool = False):
        """Clear server errors.
        
        Args:
            include_unresolved: Whether to clear unresolved errors
        """
        async with self._lock:
            if include_unresolved:
                self.server_errors.clear()
            else:
                self.server_errors = [
                    error for error in self.server_errors
                    if not error.context.get("resolved", False)
                ]
                
    async def get_error_stats(self) -> Dict[str, Any]:
        """Get server error statistics.
        
        Returns:
            Dictionary with error statistics
        """
        async with self._lock:
            total_errors = len(self.server_errors)
            resolved_errors = sum(
                1 for error in self.server_errors
                if error.context.get("resolved", False)
            )
            unresolved_errors = total_errors - resolved_errors
            recoverable_errors = sum(
                1 for error in self.server_errors
                if error.context.get("recoverable", True)
            )
            
            return {
                "total_errors": total_errors,
                "resolved_errors": resolved_errors,
                "unresolved_errors": unresolved_errors,
                "recoverable_errors": recoverable_errors,
                "error_types": {
                    error_type.value: sum(
                        1 for error in self.server_errors
                        if error.error_type == error_type
                    )
                    for error_type in AnalysisErrorType
                }
            }

    async def cleanup(self):
        """Clean up server error handler resources."""
        try:
            async with self._lock:
                self.server_errors.clear()
        except Exception as e:
            logger.error(f"Error cleaning up server error handler: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()