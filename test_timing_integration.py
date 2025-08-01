#!/usr/bin/env python3
"""
Test script to validate timing integration without requiring full CrewAI execution.
This tests the timing infrastructure and data flow end-to-end.
"""

import sys
import os
import time
import json
from typing import Dict, Any, List

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    from agents import AgentTimingTracker, TicketSummarizer
    from langsmith_integration import CrewAILangSmithHandler, get_langsmith_handler
    from database_service import DatabaseService
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_timing_tracker():
    """Test AgentTimingTracker functionality"""
    print("\n🧪 Testing AgentTimingTracker...")
    
    tracker = AgentTimingTracker()
    ticket_id = "test-ticket-123"
    
    # Test agent timing recording
    tracker.record_agent_start('triage_specialist', ticket_id)
    time.sleep(0.1)  # Simulate processing
    tracker.record_agent_end('triage_specialist', ticket_id)
    
    # Test retrieval
    duration = tracker.get_agent_duration('triage_specialist', ticket_id)
    print(f"📊 Triage specialist duration: {duration:.3f}s")
    
    if duration > 0.05 and duration < 0.5:  # Should be around 0.1s
        print("✅ AgentTimingTracker working correctly")
        return True
    else:
        print(f"❌ AgentTimingTracker duration invalid: {duration}")
        return False

def test_callback_handler():
    """Test LangSmith callback handler timing"""
    print("\n🧪 Testing CrewAILangSmithHandler...")
    
    handler = CrewAILangSmithHandler()
    
    # Simulate LLM execution
    run_id = "test-run-123"
    serialized = {"_type": "openai-chat"}
    prompts = ["Test prompt"]
    
    # Set context
    from langsmith_integration import _current_agent_name, _current_ticket_id
    token1 = _current_ticket_id.set("test-ticket-123")
    token2 = _current_agent_name.set("triage_specialist")
    
    try:
        # Start LLM
        handler.on_llm_start(serialized, prompts, run_id=run_id)
        time.sleep(0.05)  # Simulate processing
        
        # Mock LLM result
        from langchain.schema import LLMResult, Generation
        generations = [[Generation(text="Test response")]]
        llm_result = LLMResult(generations=generations)
        
        # End LLM
        handler.on_llm_end(llm_result, run_id=run_id)
        
        # Check duration tracking
        durations = handler.get_agent_durations()
        print(f"📊 Agent durations: {durations}")
        
        if 'triage_specialist' in durations and durations['triage_specialist'] > 0:
            print("✅ Callback handler timing working correctly")
            return True
        else:
            print("❌ Callback handler timing not working")
            return False
            
    finally:
        # Reset context
        _current_ticket_id.reset(token1)
        _current_agent_name.reset(token2)

def test_individual_logs_creation():
    """Test individual agent logs creation with timing data"""
    print("\n🧪 Testing individual agent logs creation...")
    
    # Create mock summarizer instance
    summarizer = TicketSummarizer()
    
    # Create timing tracker with test data
    tracker = AgentTimingTracker()
    ticket_id = "test-ticket-456"
    
    # Record some timing data
    tracker.record_agent_start('ticket_analyst', ticket_id)
    time.sleep(0.08)
    tracker.record_agent_end('ticket_analyst', ticket_id)
    
    # Create mock crew result
    class MockTaskOutput:
        def __init__(self, result, agent_name):
            self.result = result
            self.agent = type('Agent', (), {'role': agent_name})()
    
    class MockCrewResult:
        def __init__(self):
            self.tasks_output = [
                MockTaskOutput("Analysis complete", "ticket_analyst")
            ]
    
    # Create mock run info with activities
    run_info = {
        'agent_activities': {
            'ticket_analyst': [
                {
                    'run_id': 'test-run-456',
                    'agent_name': 'ticket_analyst',
                    'event_type': 'llm_start',
                    'timestamp': time.time() - 0.08
                },
                {
                    'run_id': 'test-run-456', 
                    'agent_name': 'ticket_analyst',
                    'event_type': 'llm_end',
                    'timestamp': time.time(),
                    'duration': 0.08,
                    'agent_total_duration': 0.08
                }
            ]
        }
    }
    
    # Set timing tracker on summarizer
    summarizer._current_timing_tracker = tracker
    
    # Test the method
    logs = summarizer._extract_individual_agent_activities_from_handler(
        MockCrewResult(), ticket_id, "Test ticket content", run_info, tracker
    )
    
    print(f"📊 Generated {len(logs)} individual logs")
    for log in logs:
        processing_time = log.get('processing_time', 0.0)
        agent_name = log.get('agent_name', 'unknown')
        print(f"📊 {agent_name}: {processing_time:.3f}s")
        
        if processing_time > 0:
            print(f"✅ Agent {agent_name} has non-zero processing time")
        else:
            print(f"❌ Agent {agent_name} has zero processing time")
    
    # Check if any logs have non-zero processing time
    has_timing = any(log.get('processing_time', 0.0) > 0 for log in logs)
    return has_timing

def test_database_integration():
    """Test database service processing time saving"""
    print("\n🧪 Testing database integration...")
    
    try:
        db_service = DatabaseService()
        
        # Test data
        test_log = {
            'agent_name': 'test_agent',
            'input_data': {'test': 'input'},
            'output_data': {'test': 'output'},
            'metadata': {'test': 'metadata'},
            'processing_time': 1.234,  # Non-zero processing time
            'status': 'success',
            'trace_id': 'test-trace-789'
        }
        
        # This would normally save to database - just test the method signature
        print(f"📊 Testing database call with processing_time: {test_log['processing_time']}")
        
        # Verify the method exists and accepts processing_time parameter
        method = getattr(db_service, 'save_processing_log_with_agent_stats')
        import inspect
        sig = inspect.signature(method)
        
        if 'processing_time' in sig.parameters:
            print("✅ Database service accepts processing_time parameter")
            return True
        else:
            print("❌ Database service missing processing_time parameter")
            return False
            
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False

def main():
    """Run all timing integration tests"""
    print("🚀 Starting timing integration tests...\n")
    
    tests = [
        ("AgentTimingTracker", test_timing_tracker),
        ("Callback Handler", test_callback_handler), 
        ("Individual Logs", test_individual_logs_creation),
        ("Database Integration", test_database_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("🧪 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20s} {status}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All timing integration tests PASSED!")
        print("✅ Database calls should now receive actual processing times > 0.0")
        return 0
    else:
        print(f"\n⚠️ {len(results) - passed} tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())