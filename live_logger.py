"""
Live debug logging system for Streamlit with persistent state and process control.
"""

import threading
import time
import queue
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"

class ProcessStatus(Enum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

@dataclass
class LogEntry:
    timestamp: str
    level: LogLevel
    message: str
    process_id: str
    thread_id: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'level': self.level.value,
            'message': self.message,
            'process_id': self.process_id,
            'thread_id': self.thread_id,
            'details': self.details or {}
        }

@dataclass
class ProcessInfo:
    process_id: str
    name: str
    status: ProcessStatus
    start_time: str
    progress: float = 0.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    
    def to_dict(self):
        return {
            'process_id': self.process_id,
            'name': self.name,
            'status': self.status.value,
            'start_time': self.start_time,
            'progress': self.progress,
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps
        }

class LiveLogger:
    """Thread-safe live logger with persistent state for Streamlit."""
    
    def __init__(self, max_logs: int = 1000):
        self._logs: List[LogEntry] = []
        self._processes: Dict[str, ProcessInfo] = {}
        self._log_queue = queue.Queue()
        self._max_logs = max_logs
        self._lock = threading.RLock()
        self._running = True
        self._control_signals: Dict[str, ProcessStatus] = {}
        
        # Start background log processor
        self._processor_thread = threading.Thread(target=self._process_logs, daemon=True)
        self._processor_thread.start()
    
    def _process_logs(self):
        """Background thread to process log entries."""
        while self._running:
            try:
                log_entry = self._log_queue.get(timeout=1)
                with self._lock:
                    self._logs.append(log_entry)
                    # Keep only the most recent logs
                    if len(self._logs) > self._max_logs:
                        self._logs = self._logs[-self._max_logs:]
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logger error: {e}")
    
    def log(self, level: LogLevel, message: str, process_id: str = "system", 
            thread_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Add a log entry."""
        if thread_id is None:
            thread_id = str(threading.get_ident())
        
        log_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            process_id=process_id,
            thread_id=thread_id,
            details=details
        )
        
        try:
            self._log_queue.put_nowait(log_entry)
        except queue.Full:
            # If queue is full, drop oldest and add new
            try:
                self._log_queue.get_nowait()
                self._log_queue.put_nowait(log_entry)
            except queue.Empty:
                pass
    
    def start_process(self, name: str, total_steps: int = 0) -> str:
        """Start a new tracked process."""
        process_id = str(uuid.uuid4())[:8]
        
        process_info = ProcessInfo(
            process_id=process_id,
            name=name,
            status=ProcessStatus.RUNNING,
            start_time=datetime.now().isoformat(),
            total_steps=total_steps
        )
        
        with self._lock:
            self._processes[process_id] = process_info
        
        self.log(LogLevel.INFO, f"Started process: {name}", process_id)
        return process_id
    
    def update_process(self, process_id: str, current_step: str = "", 
                      progress: float = None, completed_steps: int = None):
        """Update process information."""
        with self._lock:
            if process_id in self._processes:
                process = self._processes[process_id]
                if current_step:
                    process.current_step = current_step
                if progress is not None:
                    process.progress = progress
                if completed_steps is not None:
                    process.completed_steps = completed_steps
                    if process.total_steps > 0:
                        process.progress = completed_steps / process.total_steps
    
    def complete_process(self, process_id: str, success: bool = True):
        """Mark a process as completed."""
        with self._lock:
            if process_id in self._processes:
                self._processes[process_id].status = ProcessStatus.COMPLETED if success else ProcessStatus.ERROR
                self._processes[process_id].progress = 1.0
        
        level = LogLevel.SUCCESS if success else LogLevel.ERROR
        self.log(level, f"Process completed: {success}", process_id)
    
    def pause_process(self, process_id: str):
        """Pause a process."""
        with self._lock:
            if process_id in self._processes:
                self._processes[process_id].status = ProcessStatus.PAUSED
                self._control_signals[process_id] = ProcessStatus.PAUSED
        
        self.log(LogLevel.WARNING, "Process paused", process_id)
    
    def resume_process(self, process_id: str):
        """Resume a paused process."""
        with self._lock:
            if process_id in self._processes:
                self._processes[process_id].status = ProcessStatus.RUNNING
                if process_id in self._control_signals:
                    del self._control_signals[process_id]
        
        self.log(LogLevel.INFO, "Process resumed", process_id)
    
    def stop_process(self, process_id: str):
        """Stop a process."""
        with self._lock:
            if process_id in self._processes:
                self._processes[process_id].status = ProcessStatus.STOPPED
                self._control_signals[process_id] = ProcessStatus.STOPPED
        
        self.log(LogLevel.WARNING, "Process stopped", process_id)
    
    def should_pause(self, process_id: str) -> bool:
        """Check if process should be paused."""
        with self._lock:
            return self._control_signals.get(process_id) == ProcessStatus.PAUSED
    
    def should_stop(self, process_id: str) -> bool:
        """Check if process should be stopped."""
        with self._lock:
            return self._control_signals.get(process_id) == ProcessStatus.STOPPED
    
    def get_logs(self, process_id: Optional[str] = None, 
                 level_filter: Optional[LogLevel] = None, 
                 limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get logs with optional filtering."""
        with self._lock:
            logs = self._logs.copy()
        
        # Filter by process_id
        if process_id:
            logs = [log for log in logs if log.process_id == process_id]
        
        # Filter by level
        if level_filter:
            logs = [log for log in logs if log.level == level_filter]
        
        # Apply limit
        if limit:
            logs = logs[-limit:]
        
        return [log.to_dict() for log in logs]
    
    def get_processes(self) -> List[Dict[str, Any]]:
        """Get all process information."""
        with self._lock:
            return [process.to_dict() for process in self._processes.values()]
    
    def get_active_processes(self) -> List[Dict[str, Any]]:
        """Get only active (running or paused) processes."""
        with self._lock:
            active = [p for p in self._processes.values() 
                     if p.status in [ProcessStatus.RUNNING, ProcessStatus.PAUSED]]
            return [process.to_dict() for process in active]
    
    def clear_completed_processes(self):
        """Remove completed and stopped processes."""
        with self._lock:
            self._processes = {
                pid: process for pid, process in self._processes.items()
                if process.status not in [ProcessStatus.COMPLETED, ProcessStatus.STOPPED, ProcessStatus.ERROR]
            }
    
    def shutdown(self):
        """Shutdown the logger."""
        self._running = False
        if self._processor_thread.is_alive():
            self._processor_thread.join(timeout=2)

# Global logger instance
live_logger = LiveLogger()

# Convenience functions
def log_debug(message: str, process_id: str = "system", **kwargs):
    live_logger.log(LogLevel.DEBUG, message, process_id, **kwargs)

def log_info(message: str, process_id: str = "system", **kwargs):
    live_logger.log(LogLevel.INFO, message, process_id, **kwargs)

def log_warning(message: str, process_id: str = "system", **kwargs):
    live_logger.log(LogLevel.WARNING, message, process_id, **kwargs)

def log_error(message: str, process_id: str = "system", **kwargs):
    live_logger.log(LogLevel.ERROR, message, process_id, **kwargs)

def log_success(message: str, process_id: str = "system", **kwargs):
    live_logger.log(LogLevel.SUCCESS, message, process_id, **kwargs)