"""
LangSmith Integration Module

Provides proper LangSmith tracing integration with CrewAI multi-agent workflows.
This module replaces manual run creation with proper callback handlers that integrate
with LangChain's built-in tracing capabilities.
"""

import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
from contextvars import ContextVar

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.schema import LLMResult
from langsmith import Client


# Context variables for tracking execution state
_current_ticket_id: ContextVar[Optional[str]] = ContextVar('current_ticket_id', default=None)
_current_agent_name: ContextVar[Optional[str]] = ContextVar('current_agent_name', default=None)
_run_hierarchy: ContextVar[Dict[str, Any]] = ContextVar('run_hierarchy', default={})


class CrewAILangSmithHandler(BaseCallbackHandler):
    """
    Proper LangSmith callback handler that integrates with CrewAI execution flow.
    
    This handler tracks agent activities without interfering with LangChain's
    built-in tracing system. It captures run information for session state
    while letting LangSmith handle the actual trace lifecycle.
    
    Enhanced with accurate timing tracking for agent performance analytics.
    """
    
    def __init__(self):
        super().__init__()
        self.agent_activities: List[Dict[str, Any]] = []
        self.run_times: Dict[str, float] = {}
        self.agent_durations: Dict[str, float] = {}  # Track durations by agent
        self.agent_run_mapping: Dict[str, str] = {}  # Map run_id to agent_name
        self.client = None
        
    def _get_client(self) -> Optional[Client]:
        """Get or create LangSmith client lazily."""
        if self.client is None and os.environ.get("LANGCHAIN_TRACING_V2") == "true":
            try:
                self.client = Client()
            except Exception as e:
                print(f"Warning: Could not initialize LangSmith client: {e}")
        return self.client
    
    def on_llm_start(
        self, 
        serialized: Dict[str, Any], 
        prompts: List[str], 
        **kwargs: Any
    ) -> Any:
        """Called when LLM starts running."""
        run_id = kwargs.get('run_id')
        if run_id:
            start_time = time.time()
            self.run_times[str(run_id)] = start_time
            
            # Track agent activity for session state
            ticket_id = _current_ticket_id.get()
            agent_name = _current_agent_name.get()
            
            # Try to extract agent information from serialized data
            if not agent_name:
                # Look for agent information in the serialized data
                if 'name' in serialized:
                    agent_name = serialized['name']
                elif 'id' in serialized:
                    # Try to extract agent name from ID
                    agent_id = serialized['id']
                    if 'triage' in str(agent_id).lower():
                        agent_name = 'triage_specialist'
                    elif 'analyst' in str(agent_id).lower():
                        agent_name = 'ticket_analyst'
                    elif 'strategist' in str(agent_id).lower():
                        agent_name = 'support_strategist'
                    elif 'qa' in str(agent_id).lower() or 'reviewer' in str(agent_id).lower():
                        agent_name = 'qa_reviewer'
            
            if agent_name:
                # Map run_id to agent for duration calculation
                self.agent_run_mapping[str(run_id)] = agent_name
                
                activity = {
                    'run_id': str(run_id),
                    'agent_name': agent_name,
                    'ticket_id': ticket_id,
                    'event_type': 'llm_start',
                    'timestamp': start_time,
                    'model_name': serialized.get('_type', 'unknown'),
                    'prompts': prompts[:1] if prompts else [],  # Only store first prompt for brevity
                    'serialized_info': {
                        'name': serialized.get('name', 'unknown'),
                        'id': serialized.get('id', 'unknown'),
                        'type': serialized.get('_type', 'unknown')
                    }
                }
                self.agent_activities.append(activity)
                print(f"🔗 LangSmith: {agent_name} LLM started (run_id: {run_id})")
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Called when LLM ends running."""
        run_id = kwargs.get('run_id')
        if run_id:
            start_time = self.run_times.pop(str(run_id), time.time())
            end_time = time.time()
            duration = end_time - start_time
            
            # Track completion for session state
            ticket_id = _current_ticket_id.get()
            agent_name = _current_agent_name.get() or self.agent_run_mapping.get(str(run_id))
            
            if agent_name:
                # Accumulate duration for this agent
                if agent_name not in self.agent_durations:
                    self.agent_durations[agent_name] = 0.0
                self.agent_durations[agent_name] += duration
                
                # Clean up mapping
                self.agent_run_mapping.pop(str(run_id), None)
                
                # Safely extract response information
                token_usage = {}
                response_count = 0
                
                if response:
                    token_usage = getattr(response, 'llm_output', {}).get('token_usage', {})
                    if hasattr(response, 'generations') and response.generations:
                        response_count = len(response.generations)
                
                activity = {
                    'run_id': str(run_id),
                    'agent_name': agent_name,
                    'ticket_id': ticket_id,
                    'event_type': 'llm_end',
                    'timestamp': end_time,
                    'duration': duration,
                    'token_usage': token_usage,
                    'response_count': response_count,
                    'agent_total_duration': self.agent_durations[agent_name]  # Track cumulative time
                }
                self.agent_activities.append(activity)
                print(f"🔗 LangSmith: {agent_name} LLM completed in {duration:.2f}s (total: {self.agent_durations[agent_name]:.2f}s)")
    
    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Called when LLM encounters an error."""
        run_id = kwargs.get('run_id')
        if run_id:
            self.run_times.pop(str(run_id), None)
            
            # Track error for session state
            ticket_id = _current_ticket_id.get()
            agent_name = _current_agent_name.get()
            
            if agent_name:
                activity = {
                    'run_id': str(run_id),
                    'agent_name': agent_name,
                    'ticket_id': ticket_id,
                    'event_type': 'llm_error',
                    'timestamp': time.time(),
                    'error': str(error),
                    'error_type': type(error).__name__
                }
                self.agent_activities.append(activity)
    
    def on_chain_start(
        self, 
        serialized: Dict[str, Any], 
        inputs: Dict[str, Any], 
        **kwargs: Any
    ) -> Any:
        """Called when chain starts running."""
        run_id = kwargs.get('run_id')
        if run_id:
            self.run_times[str(run_id)] = time.time()
            
            # Track chain activity for CrewAI agent chains
            ticket_id = _current_ticket_id.get()
            agent_name = _current_agent_name.get()
            
            if agent_name and 'agent' in serialized.get('_type', '').lower():
                activity = {
                    'run_id': str(run_id),
                    'agent_name': agent_name,
                    'ticket_id': ticket_id,
                    'event_type': 'agent_start',
                    'timestamp': time.time(),
                    'chain_type': serialized.get('_type', 'unknown'),
                    'inputs': {k: str(v)[:200] for k, v in inputs.items()}  # Truncate inputs
                }
                self.agent_activities.append(activity)
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Called when chain ends running."""
        run_id = kwargs.get('run_id')
        if run_id:
            start_time = self.run_times.pop(str(run_id), time.time())
            duration = time.time() - start_time
            
            # Track completion for agent chains
            ticket_id = _current_ticket_id.get()
            agent_name = _current_agent_name.get()
            
            if agent_name:
                activity = {
                    'run_id': str(run_id),
                    'agent_name': agent_name,
                    'ticket_id': ticket_id,
                    'event_type': 'agent_end',
                    'timestamp': time.time(),
                    'duration': duration,
                    'outputs': {k: str(v)[:200] for k, v in outputs.items()}  # Truncate outputs
                }
                self.agent_activities.append(activity)
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Called when chain encounters an error."""
        run_id = kwargs.get('run_id')
        if run_id:
            self.run_times.pop(str(run_id), None)
            
            # Track error for agent chains
            ticket_id = _current_ticket_id.get()
            agent_name = _current_agent_name.get()
            
            if agent_name:
                activity = {
                    'run_id': str(run_id),
                    'agent_name': agent_name,
                    'ticket_id': ticket_id,
                    'event_type': 'agent_error',
                    'timestamp': time.time(),
                    'error': str(error),
                    'error_type': type(error).__name__
                }
                self.agent_activities.append(activity)
    
    def get_agent_activities(self) -> List[Dict[str, Any]]:
        """Get all captured agent activities."""
        return self.agent_activities.copy()
    
    def get_agent_durations(self) -> Dict[str, float]:
        """Get accumulated durations by agent name."""
        return self.agent_durations.copy()
    
    def clear_activities(self) -> None:
        """Clear captured activities and timing data."""
        self.agent_activities.clear()
        self.run_times.clear()
        self.agent_durations.clear()
        self.agent_run_mapping.clear()


# Global handler instance
_langsmith_handler = CrewAILangSmithHandler()


def get_langsmith_handler() -> CrewAILangSmithHandler:
    """Get the global LangSmith handler instance."""
    return _langsmith_handler


@contextmanager
def langsmith_context(ticket_id: str, agent_name: Optional[str] = None):
    """
    Context manager for setting LangSmith tracing context.
    
    Args:
        ticket_id: The ticket ID being processed
        agent_name: The current agent name (optional)
    """
    # Set context variables
    ticket_token = _current_ticket_id.set(ticket_id)
    agent_token = _current_agent_name.set(agent_name) if agent_name else None
    
    try:
        yield
    finally:
        # Reset context variables
        _current_ticket_id.reset(ticket_token)
        if agent_token:
            _current_agent_name.reset(agent_token)


def setup_langsmith_tracing() -> bool:
    """
    Set up LangSmith tracing environment properly.
    
    This function ensures LangSmith environment variables are set correctly
    and validates the configuration without creating manual runs.
    
    Returns:
        bool: True if LangSmith tracing is enabled and configured
    """
    try:
        # Check if LangSmith is configured
        api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        if not api_key:
            print("ℹ️ LangSmith tracing disabled (no API key)")
            return False
        
        # Set required environment variables for LangChain tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        
        # Set project name if not already set
        if not os.environ.get("LANGCHAIN_PROJECT"):
            os.environ["LANGCHAIN_PROJECT"] = "ticket-sum"
        
        # Set endpoint if not already set
        if not os.environ.get("LANGCHAIN_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        
        # Test client connection
        client = Client()
        
        print(f"✅ LangSmith tracing enabled")
        print(f"📡 Project: {os.environ.get('LANGCHAIN_PROJECT')}")
        print(f"🔗 Endpoint: {os.environ.get('LANGCHAIN_ENDPOINT')}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ LangSmith setup failed: {e}")
        # Disable tracing on failure
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False


def create_callback_manager() -> CallbackManager:
    """
    Create a CallbackManager with proper LangSmith integration.
    
    This replaces manual callback addition with proper CallbackManager usage
    that CrewAI will respect.
    
    Returns:
        CallbackManager: Configured callback manager
    """
    handlers = []
    
    # Add our custom handler for activity tracking
    handlers.append(_langsmith_handler)
    
    # LangChain will automatically add LangSmith handler if tracing is enabled
    return CallbackManager(handlers)


def get_run_information() -> Dict[str, Any]:
    """
    Get captured run information from the handler.
    
    This replaces manual run ID capture with proper callback-based tracking.
    
    Returns:
        Dict containing run information and agent activities
    """
    activities = _langsmith_handler.get_agent_activities()
    
    # Extract unique run IDs
    run_ids = list(set(activity['run_id'] for activity in activities if 'run_id' in activity))
    
    # Group activities by agent
    agent_activities = {}
    for activity in activities:
        agent_name = activity.get('agent_name', 'unknown')
        if agent_name not in agent_activities:
            agent_activities[agent_name] = []
        agent_activities[agent_name].append(activity)
    
    return {
        'run_ids': run_ids,
        'agent_activities': agent_activities,
        'total_activities': len(activities),
        'unique_agents': list(agent_activities.keys())
    }


def clear_run_information() -> None:
    """Clear captured run information."""
    _langsmith_handler.clear_activities()


def submit_completion_metadata(ticket_id: str, metadata: Dict[str, Any]) -> None:
    """
    Submit completion metadata to LangSmith.
    
    This uses the proper LangSmith client to add metadata to completed runs
    instead of creating new orphaned runs.
    
    Args:
        ticket_id: The processed ticket ID
        metadata: Metadata to attach to the runs
    """
    try:
        client = _langsmith_handler._get_client()
        if not client:
            return
        
        activities = _langsmith_handler.get_agent_activities()
        run_ids = list(set(activity['run_id'] for activity in activities if 'run_id' in activity))
        
        completion_metadata = {
            "ticket_id": ticket_id,
            "processing_type": "collaborative_multi_agent",
            "completion_status": "completed",
            "system_version": "v2.0",
            **metadata
        }
        
        # Add metadata to each run
        for run_id in run_ids:
            try:
                client.update_run(
                    run_id=run_id,
                    extra=completion_metadata
                )
                
                # Add completion feedback
                client.create_feedback(
                    run_id=run_id,
                    key="completion_status",
                    score=1.0,
                    comment=f"Multi-agent processing completed for ticket {ticket_id}"
                )
                
            except Exception as e:
                print(f"Warning: Could not update run {run_id}: {e}")
        
        if run_ids:
            print(f"📡 Updated {len(run_ids)} LangSmith runs with completion metadata")
            
    except Exception as e:
        print(f"Warning: Could not submit completion metadata: {e}")