"""Async task manager for topic analysis pipeline."""
from typing import Optional, Dict, Any, Callable, Awaitable, Union, List, Set
import asyncio
from contextlib import asynccontextmanager
from loguru import logger
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from pathlib import Path
import json
import aiofiles
import os

from .async_utils import AsyncRunner, TaskStatus, TaskInfo
from .error_handlers import ErrorHandler, AnalysisErrorType

@dataclass
class AnalysisTaskInfo:
    """Enhanced task information for topic analysis."""
    id: str
    start_time: datetime
    status: TaskStatus
    model_type: str
    num_topics: int
    doc_count: int
    language_stats: Dict[str, int] = field(default_factory=dict)
    progress: float = 0.0
    message: Optional[str] = None
    result: Any = None
    error: Optional[Dict] = None
    end_time: Optional[datetime] = None
    model_params: Dict[str, Any] = field(default_factory=dict)
    coherence_scores: List[float] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class AsyncTaskManager:
    """Enhanced async task manager for topic analysis."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_concurrent_tasks: int = 3,
        memory_threshold: float = 0.9,
        cpu_threshold: float = 0.9
    ):
        """Initialize task manager.
        
        Args:
            cache_dir: Directory for task caching
            max_concurrent_tasks: Maximum concurrent analysis tasks
            memory_threshold: Maximum memory usage threshold
            cpu_threshold: Maximum CPU usage threshold
        """
        self.cache_dir = cache_dir or Path('cache/tasks')
        self._tasks: Dict[str, AnalysisTaskInfo] = {}
        self._active_tasks: Set[str] = set()
        self._runner = AsyncRunner(
            memory_threshold=memory_threshold,
            cpu_threshold=cpu_threshold
        )
        self._lock = asyncio.Lock()
        self._task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._shutdown = False
        self._cleanup_task = None
        self._error_handler = ErrorHandler()
        self._periodic_save_task = None
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"AsyncTaskManager initialized (max tasks: {max_concurrent_tasks}, "
            f"memory threshold: {memory_threshold}, CPU threshold: {cpu_threshold})"
        )

    async def start_analysis_task(
        self,
        task_id: str,
        texts: List[str],
        model_type: str = 'lda',
        num_topics: int = 20,
        model_params: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> AnalysisTaskInfo:
        """Start a new topic analysis task.
        
        Args:
            task_id: Task identifier
            texts: Documents to analyze
            model_type: Analysis method
            num_topics: Number of topics
            model_params: Additional model parameters
            priority: Task priority
            
        Returns:
            Task information
            
        Raises:
            ValueError: If task_id already exists
            ResourceWarning: If system resources are insufficient
        """
        try:
            # Check if task exists
            async with self._lock:
                if task_id in self._tasks:
                    raise ValueError(f"Task {task_id} already exists")
                
                # Create task info
                task_info = AnalysisTaskInfo(
                    id=task_id,
                    start_time=datetime.now(),
                    status=TaskStatus.PENDING,
                    model_type=model_type,
                    num_topics=num_topics,
                    doc_count=len(texts),
                    model_params=model_params or {}
                )
                self._tasks[task_id] = task_info
            
            # Define progress callback
            async def update_progress(progress: float, message: str):
                await self._update_task_status(
                    task_id,
                    progress,
                    message,
                    measure_performance=True
                )
            
            # Run analysis task with semaphore
            async with self._task_semaphore:
                self._active_tasks.add(task_id)
                try:
                    await self._update_task_status(
                        task_id,
                        0.0,
                        "Starting analysis",
                        status=TaskStatus.RUNNING
                    )
                    
                    # Run analysis through runner
                    result = await self._runner.run_with_progress(
                        self._run_analysis,
                        texts=texts,
                        model_type=model_type,
                        num_topics=num_topics,
                        model_params=model_params,
                        task_id=task_id,
                        progress_callback=update_progress,
                        priority=priority
                    )
                    
                    # Update final status
                    await self._update_task_status(
                        task_id,
                        1.0,
                        "Analysis completed",
                        status=TaskStatus.COMPLETED,
                        result=result
                    )
                    
                    return task_info
                    
                except Exception as e:
                    error = await self._error_handler.capture_error(
                        e,
                        AnalysisErrorType.ANALYSIS,
                        component="task_manager"
                    )
                    await self._update_task_status(
                        task_id,
                        0.0,
                        str(error.message),
                        status=TaskStatus.FAILED,
                        error=error.to_dict()
                    )
                    raise
                finally:
                    self._active_tasks.remove(task_id)
                    
        except Exception as e:
            logger.error(f"Error starting analysis task: {e}")
            raise

    async def _run_analysis(
        self,
        texts: List[str],
        model_type: str,
        num_topics: int,
        model_params: Optional[Dict[str, Any]] = None,
        task_id: str = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Run topic analysis task.
        
        Args:
            texts: List of document texts
            model_type: Analysis method ('lda', 'nmf', or 'lsa')
            num_topics: Number of topics to extract
            model_params: Additional model parameters
            task_id: Task identifier
            progress_callback: Optional progress callback
        
        Returns:
            Analysis results dictionary
        """
        try:
            # Initialize analysis components
            from .analyzer import TopicAnalyzer
            analyzer = TopicAnalyzer(None, lang='bilingual')
            await analyzer.initialize()
            
            # Run analysis
            topics = await analyzer.analyze_topics(
                texts,
                method=model_type,
                num_topics=num_topics,
                model_params=model_params or {},
                progress_callback=progress_callback
            )
            
            # Calculate performance metrics if task info available
            performance = {}
            if task_id and task_id in self._tasks:
                task_info = self._tasks[task_id]
                performance = {
                    'time_taken': (datetime.now() - task_info.start_time).total_seconds(),
                    'docs_per_second': len(texts) / max(1, (datetime.now() - task_info.start_time).total_seconds()),
                    'memory_used_mb': task_info.performance_metrics.get('memory_usage', 0) * 1024,
                    'cpu_usage_percent': task_info.performance_metrics.get('cpu_usage', 0) * 100
                }
            
            return {
                'topics': topics,
                'performance': performance,
                'model_info': {
                    'type': model_type,
                    'num_topics': num_topics,
                    'params': model_params or {}
                }
            }
            
        except Exception as e:
            logger.error(f"Error in analysis task: {e}")
            raise
        finally:
            if analyzer:
                await analyzer.cleanup()

    async def start_analysis_task(
        self,
        task_id: str,
        texts: List[str],
        model_type: str = 'lda',
        num_topics: int = 20,
        model_params: Optional[Dict[str, Any]] = None,
        priority: int = 0
    ) -> 'AnalysisTaskInfo':
        """Start a new topic analysis task.
        
        Args:
            task_id: Task identifier
            texts: Documents to analyze
            model_type: Analysis method
            num_topics: Number of topics
            model_params: Additional model parameters
            priority: Task priority
            
        Returns:
            Task information
        """
        try:
            # Check if task exists
            async with self._lock:
                if task_id in self._tasks:
                    raise ValueError(f"Task {task_id} already exists")
                
                # Create task info
                task_info = AnalysisTaskInfo(
                    id=task_id,
                    start_time=datetime.now(),
                    status=TaskStatus.PENDING,
                    model_type=model_type,
                    num_topics=num_topics,
                    doc_count=len(texts),
                    model_params=model_params or {}
                )
                self._tasks[task_id] = task_info
            
            # Define progress callback
            async def update_progress(progress: float, message: str):
                await self._update_task_status(
                    task_id,
                    progress,
                    message,
                    measure_performance=True
                )
            
            # Run analysis task with semaphore
            async with self._task_semaphore:
                self._active_tasks.add(task_id)
                try:
                    await self._update_task_status(
                        task_id,
                        0.0,
                        "Starting analysis",
                        status=TaskStatus.RUNNING
                    )
                    
                    # Run analysis through runner
                    result = await self._run_analysis(
                        texts=texts,
                        model_type=model_type,
                        num_topics=num_topics,
                        model_params=model_params,
                        task_id=task_id,
                        progress_callback=update_progress
                    )
                    
                    # Update final status
                    await self._update_task_status(
                        task_id,
                        1.0,
                        "Analysis completed",
                        status=TaskStatus.COMPLETED,
                        result=result
                    )
                    
                    return task_info
                    
                except Exception as e:
                    error = await self._error_handler.capture_error(
                        e,
                        AnalysisErrorType.ANALYSIS,
                        component="task_manager"
                    )
                    await self._update_task_status(
                        task_id,
                        0.0,
                        str(error.message),
                        status=TaskStatus.FAILED,
                        error=error.to_dict()
                    )
                    raise
                finally:
                    self._active_tasks.remove(task_id)
                    
        except Exception as e:
            logger.error(f"Error starting analysis task: {e}")
            raise

    async def _update_task_status(
        self,
        task_id: str,
        progress: float,
        message: str,
        status: Optional[TaskStatus] = None,
        result: Any = None,
        error: Optional[Dict] = None,
        measure_performance: bool = False
    ):
        """Update task status with performance metrics."""
        try:
            async with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.progress = progress
                    task.message = message
                    
                    if status:
                        task.status = status
                    if result is not None:
                        task.result = result
                    if error:
                        task.error = error
                    
                    # Update performance metrics if requested
                    if measure_performance:
                        import psutil
                        process = psutil.Process()
                        task.performance_metrics.update({
                            'memory_usage': process.memory_percent() / 100,
                            'cpu_usage': process.cpu_percent() / 100,
                            'duration': (datetime.now() - task.start_time).total_seconds()
                        })
                    
                    if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                        task.end_time = datetime.now()
                        
                    # Save task state
                    await self._save_task_state(task_id)
                    
        except Exception as e:
            logger.error(f"Error updating task status: {e}")

    async def get_task_status(
        self,
        task_id: str
    ) -> Optional[AnalysisTaskInfo]:
        """Get current status of a task."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        include_completed: bool = True,
        max_age: Optional[timedelta] = None
    ) -> List[AnalysisTaskInfo]:
        """List tasks with filtering options."""
        async with self._lock:
            tasks = list(self._tasks.values())
            
        if status:
            tasks = [task for task in tasks if task.status == status]
            
        if not include_completed:
            tasks = [
                task for task in tasks 
                if task.status not in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED
                }
            ]
            
        if max_age:
            cutoff = datetime.now() - max_age
            tasks = [
                task for task in tasks
                if task.start_time > cutoff
            ]
            
        return tasks

    async def cancel_task(self, task_id: str):
        """Cancel a running task."""
        async with self._lock:
            if task_id in self._tasks and task_id in self._active_tasks:
                await self._update_task_status(
                    task_id,
                    self._tasks[task_id].progress,
                    "Task cancelled by user",
                    status=TaskStatus.CANCELLED
                )

    async def start_cleanup_task(self, interval: int = 3600):
        """Start periodic cleanup task."""
        async def periodic_cleanup():
            while not self._shutdown:
                await self.cleanup_tasks()
                await asyncio.sleep(interval)
                
        self._cleanup_task = asyncio.create_task(periodic_cleanup())
        
        # Start periodic state saving
        async def periodic_save():
            while not self._shutdown:
                await self._save_all_tasks()
                await asyncio.sleep(300)  # Save every 5 minutes
                
        self._periodic_save_task = asyncio.create_task(periodic_save())

    async def cleanup_tasks(self, max_age: timedelta = timedelta(days=7)):
        """Clean up old tasks."""
        cutoff = datetime.now() - max_age
        async with self._lock:
            self._tasks = {
                task_id: task
                for task_id, task in self._tasks.items()
                if (task.status == TaskStatus.RUNNING or
                    task.start_time > cutoff)
            }

    async def _save_task_state(self, task_id: str):
        """Save task state to cache."""
        try:
            task = self._tasks[task_id]
            cache_file = self.cache_dir / f"{task_id}.json"
            
            # Convert task to dictionary
            task_dict = {
                'id': task.id,
                'start_time': task.start_time.isoformat(),
                'status': task.status.value,
                'model_type': task.model_type,
                'num_topics': task.num_topics,
                'doc_count': task.doc_count,
                'language_stats': task.language_stats,
                'progress': task.progress,
                'message': task.message,
                'performance_metrics': task.performance_metrics
            }
            
            if task.end_time:
                task_dict['end_time'] = task.end_time.isoformat()
            
            async with aiofiles.open(cache_file, 'w') as f:
                await f.write(json.dumps(task_dict, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving task state: {e}")

    async def _save_all_tasks(self):
        """Save all task states."""
        try:
            async with self._lock:
                for task_id in self._tasks:
                    await self._save_task_state(task_id)
        except Exception as e:
            logger.error(f"Error saving all tasks: {e}")

    async def _load_task_states(self):
        """Load task states from cache."""
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    async with aiofiles.open(cache_file) as f:
                        content = await f.read()
                        task_dict = json.loads(content)
                        
                    # Convert back to TaskInfo
                    task = AnalysisTaskInfo(
                        id=task_dict['id'],
                        start_time=datetime.fromisoformat(task_dict['start_time']),
                        status=TaskStatus(task_dict['status']),
                        model_type=task_dict['model_type'],
                        num_topics=task_dict['num_topics'],
                        doc_count=task_dict['doc_count'],
                        language_stats=task_dict['language_stats'],
                        progress=task_dict['progress'],
                        message=task_dict['message'],
                        performance_metrics=task_dict['performance_metrics']
                    )
                    
                    if 'end_time' in task_dict:
                        task.end_time = datetime.fromisoformat(task_dict['end_time'])
                        
                    self._tasks[task.id] = task
                    
                except Exception as e:
                    logger.error(f"Error loading task state from {cache_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading task states: {e}")

    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up AsyncTaskManager resources")
        try:
            # Signal shutdown
            self._shutdown = True
            
            # Cancel cleanup task if running
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
                    
            # Cancel periodic save task if running
            if self._periodic_save_task and not self._periodic_save_task.done():
                self._periodic_save_task.cancel()
                try:
                    await self._periodic_save_task
                except asyncio.CancelledError:
                    pass
                    
            # Save final state
            await self._save_all_tasks()
            
            # Cancel any running tasks
            for task_id in list(self._active_tasks):
                await self.cancel_task(task_id)
            
            # Clean up runner
            await self._runner.cleanup()
            
            # Clean up error handler
            await self._error_handler.cleanup()
            
            # Clear task lists
            self._tasks.clear()
            self._active_tasks.clear()
            
            logger.info("AsyncTaskManager cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during AsyncTaskManager cleanup: {e}")
            raise

    async def get_task_metrics(self) -> Dict[str, Any]:
        """Get comprehensive task metrics."""
        try:
            metrics = {
                'total_tasks': len(self._tasks),
                'active_tasks': len(self._active_tasks),
                'completed_tasks': sum(
                    1 for task in self._tasks.values()
                    if task.status == TaskStatus.COMPLETED
                ),
                'failed_tasks': sum(
                    1 for task in self._tasks.values()
                    if task.status == TaskStatus.FAILED
                ),
                'avg_duration': 0.0,
                'avg_memory_usage': 0.0,
                'avg_cpu_usage': 0.0,
                'success_rate': 0.0
            }
            
            completed_tasks = [
                task for task in self._tasks.values()
                if task.end_time and task.start_time
            ]
            
            if completed_tasks:
                # Calculate averages
                metrics['avg_duration'] = sum(
                    (task.end_time - task.start_time).total_seconds()
                    for task in completed_tasks
                ) / len(completed_tasks)
                
                metrics['avg_memory_usage'] = sum(
                    task.performance_metrics.get('memory_usage', 0)
                    for task in completed_tasks
                ) / len(completed_tasks)
                
                metrics['avg_cpu_usage'] = sum(
                    task.performance_metrics.get('cpu_usage', 0)
                    for task in completed_tasks
                ) / len(completed_tasks)
                
                # Calculate success rate
                successful = sum(
                    1 for task in completed_tasks
                    if task.status == TaskStatus.COMPLETED
                )
                metrics['success_rate'] = successful / len(completed_tasks)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting task metrics: {e}")
            return {}

    async def get_task_history(
        self,
        max_entries: int = 100
    ) -> List[Dict[str, Any]]:
        """Get task execution history."""
        try:
            history = []
            tasks = sorted(
                [t for t in self._tasks.values() if t.end_time],
                key=lambda x: x.end_time or datetime.min,
                reverse=True
            )[:max_entries]
            
            for task in tasks:
                history.append({
                    'id': task.id,
                    'start_time': task.start_time,
                    'end_time': task.end_time,
                    'status': task.status.value,
                    'model_type': task.model_type,
                    'num_topics': task.num_topics,
                    'doc_count': task.doc_count,
                    'duration': (task.end_time - task.start_time).total_seconds()
                    if task.end_time else None,
                    'performance': task.performance_metrics
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting task history: {e}")
            return []

    async def __aenter__(self):
        """Async context manager entry."""
        await self._load_task_states()
        await self.start_cleanup_task()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()