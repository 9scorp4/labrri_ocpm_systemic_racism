"""Enhanced async utilities for topic analysis pipeline."""
import asyncio
from typing import Any, Callable, Coroutine, TypeVar, Optional, Dict, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import functools
from contextlib import contextmanager, asynccontextmanager
from loguru import logger
import time
import signal
from dataclasses import dataclass
from enum import Enum
import threading
import psutil

T = TypeVar('T')
R = TypeVar('R')

class TaskStatus(Enum):
    """Enhanced status tracking for async tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RESOURCE_LIMITED = "resource_limited"
    STALLED = "stalled"

@dataclass
class TaskInfo:
    """Enhanced task information tracking."""
    id: str
    status: TaskStatus
    start_time: float
    end_time: Optional[float] = None
    error: Optional[Exception] = None
    result: Any = None
    progress: float = 0.0
    message: Optional[str] = None
    resources: Dict[str, Any] = None
    stack_trace: Optional[str] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None

class ResourceMonitor:
    """Monitor system resources for tasks."""
    
    def __init__(self, 
                 memory_threshold: float = 0.9, 
                 cpu_threshold: float = 0.9):
        """Initialize resource monitor.
        
        Args:
            memory_threshold: Maximum memory usage threshold (0-1)
            cpu_threshold: Maximum CPU usage threshold (0-1)
        """
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self._lock = threading.Lock()

    def check_resources(self) -> Tuple[bool, Dict[str, float]]:
        """Check system resources."""
        with self._lock:
            try:
                memory = psutil.virtual_memory()
                cpu = psutil.cpu_percent(interval=0.1) / 100.0
                
                resources = {
                    'memory_percent': memory.percent / 100.0,
                    'cpu_percent': cpu,
                    'memory_available': memory.available,
                    'memory_total': memory.total
                }
                
                has_resources = (
                    resources['memory_percent'] < self.memory_threshold and
                    resources['cpu_percent'] < self.cpu_threshold
                )
                
                return has_resources, resources
                
            except Exception as e:
                logger.error(f"Error checking resources: {e}")
                return False, {}

class AsyncRunner:
    """Enhanced async runner with resource monitoring and task management."""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 memory_threshold: float = 0.9,
                 cpu_threshold: float = 0.9):
        """Initialize async runner.
        
        Args:
            max_workers: Maximum number of thread pool workers
            memory_threshold: Maximum memory usage threshold
            cpu_threshold: Maximum CPU usage threshold
        """
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._shutdown = False
        self._resource_monitor = ResourceMonitor(
            memory_threshold=memory_threshold,
            cpu_threshold=cpu_threshold
        )
        
        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self._shutdown = True
        logger.info(f"Received signal {signum}, initiating shutdown")

    async def run_async(
        self,
        func: Callable[..., T],
        *args: Any,
        task_id: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        retry_count: int = 3,
        **kwargs: Any
    ) -> T:
        """Run a synchronous function asynchronously with enhanced monitoring.
        
        Args:
            func: Function to run
            *args: Positional arguments
            task_id: Optional task identifier
            priority: Task priority (higher runs first)
            timeout: Optional timeout in seconds
            retry_count: Number of retries on failure
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            asyncio.CancelledError: If task is cancelled
            TimeoutError: If task times out
            Exception: If function raises an exception
        """
        if self._shutdown:
            raise asyncio.CancelledError("Runner is shutting down")
            
        task_id = task_id or f"task_{time.time()}"
        
        try:
            # Check resources
            has_resources, resources = self._resource_monitor.check_resources()
            if not has_resources:
                logger.warning("Insufficient system resources")
                raise ResourceWarning("Insufficient system resources")

            # Register task
            async with self._lock:
                self._tasks[task_id] = TaskInfo(
                    id=task_id,
                    status=TaskStatus.PENDING,
                    start_time=time.time(),
                    resources=resources
                )
            
            # Update status
            await self._update_task_status(
                task_id,
                TaskStatus.RUNNING,
                progress=0.0,
                message="Starting task"
            )
            
            # Run with timeout if specified
            if timeout:
                try:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._executor,
                            functools.partial(func, *args, **kwargs)
                        ),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    await self._update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        message=f"Task timed out after {timeout}s"
                    )
                    raise
            else:
                # Run normally
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    functools.partial(func, *args, **kwargs)
                )
            
            # Update final status
            await self._update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                progress=1.0,
                message="Task completed",
                result=result
            )
            
            return result
            
        except asyncio.CancelledError:
            await self._update_task_status(
                task_id,
                TaskStatus.CANCELLED,
                message="Task cancelled"
            )
            raise
        except Exception as e:
            # Retry on failure if retries remaining
            if retry_count > 0:
                logger.warning(f"Task {task_id} failed, retrying ({retry_count} retries left)")
                return await self.run_async(
                    func, *args,
                    task_id=task_id,
                    priority=priority,
                    timeout=timeout,
                    retry_count=retry_count-1,
                    **kwargs
                )
            
            await self._update_task_status(
                task_id,
                TaskStatus.FAILED,
                message=f"Task failed: {str(e)}",
                error=e
            )
            raise

    async def run_with_progress(
        self,
        func: Callable[..., T],
        *args: Any,
        task_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs: Any
    ) -> T:
        """Run a function with progress tracking.
        
        Args:
            func: Function to run
            *args: Positional arguments
            task_id: Optional task identifier
            progress_callback: Optional progress callback
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        if progress_callback:
            async def wrapped_callback(progress: float, message: str):
                await self._update_task_status(
                    task_id,
                    TaskStatus.RUNNING,
                    progress=progress,
                    message=message
                )
                await progress_callback(progress, message)
                
            kwargs['progress_callback'] = wrapped_callback
            
        return await self.run_async(func, *args, task_id=task_id, **kwargs)

    async def run_batch(
        self,
        funcs: List[Callable[..., T]],
        *args_list: List[Any],
        batch_size: int = 10,
        **kwargs_list: Dict[str, Any]
    ) -> List[T]:
        """Run multiple functions in batches.
        
        Args:
            funcs: List of functions
            *args_list: List of argument lists
            batch_size: Size of batches
            **kwargs_list: List of keyword argument dictionaries
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(funcs), batch_size):
            # Check resources before each batch
            has_resources, _ = self._resource_monitor.check_resources()
            if not has_resources:
                logger.warning("Insufficient resources, waiting before next batch")
                await asyncio.sleep(5)  # Wait before retrying
            
            batch_funcs = funcs[i:i + batch_size]
            batch_args = [args_list[j:j + batch_size] for j in range(0, len(args_list), batch_size)]
            batch_kwargs = [kwargs_list[j:j + batch_size] for j in range(0, len(kwargs_list), batch_size)]
            
            tasks = [
                self.run_async(func, *args, **kwargs)
                for func, args, kwargs in zip(batch_funcs, batch_args, batch_kwargs)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            results.extend(batch_results)
            
        return results

    async def _update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: float = 0.0,
        message: Optional[str] = None,
        result: Any = None,
        error: Optional[Exception] = None
    ):
        """Update task status with resource usage."""
        try:
            async with self._lock:
                if task_id in self._tasks:
                    task = self._tasks[task_id]
                    task.status = status
                    task.progress = progress
                    task.message = message
                    
                    if result is not None:
                        task.result = result
                    if error is not None:
                        task.error = error
                        
                    # Update resource usage
                    process = psutil.Process()
                    task.memory_usage = process.memory_percent()
                    task.cpu_usage = process.cpu_percent()
                    
                    if status in (
                        TaskStatus.COMPLETED, 
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.RESOURCE_LIMITED
                    ):
                        task.end_time = time.time()
                        
        except Exception as e:
            logger.error(f"Error updating task status: {e}")

    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get information about a task."""
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        include_completed: bool = True
    ) -> List[TaskInfo]:
        """List tasks with optional filtering."""
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
            
        return tasks

    async def cleanup_tasks(self, max_age: float = 3600):
        """Clean up old completed tasks."""
        current_time = time.time()
        async with self._lock:
            self._tasks = {
                task_id: info
                for task_id, info in self._tasks.items()
                if (info.status == TaskStatus.RUNNING or
                    (current_time - info.start_time) < max_age)
            }

    async def cleanup(self):
        """Clean up resources."""
        self._shutdown = True
        if self._executor:
            self._executor.shutdown(wait=False)
        self._tasks.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()
