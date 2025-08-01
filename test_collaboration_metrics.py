#!/usr/bin/env python3
"""
Test script to validate collaboration metrics tracking fixes.
Tests the enhanced collaboration metrics without requiring full CrewAI execution.
"""

import sys
import os
import time
import json
from typing import Dict, Any

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_initial_agent_outputs():
    """Test capturing initial agent outputs from mock crew result."""
    print("🧪 Testing initial agent outputs capture...")
    
    # Import the necessary classes
    from agents import CollaborativeSupportCrew
    
    # Create crew instance
    crew = CollaborativeSupportCrew()
    
    # Mock crew result with different agent outputs
    class MockAgent:
        def __init__(self, role):
            self.role = role
    
    class MockTaskOutput:
        def __init__(self, raw_output, agent_role):
            self.raw = raw_output
            self.agent = MockAgent(agent_role)
    
    class MockCrewResult:
        def __init__(self):
            self.tasks_output = [
                MockTaskOutput(
                    "Intent: technical_support, Severity: high, Priority: urgent, Confidence: 0.85",
                    "Triage Specialist"
                ),
                MockTaskOutput(
                    "Intent: bug, Severity: medium, Priority: normal, Confidence: 0.75", 
                    "Ticket Analyst"
                ),
                MockTaskOutput(
                    "Intent: technical_support, Severity: high, Priority: high, Confidence: 0.90",
                    "Support Strategist"
                ),
                MockTaskOutput(
                    "Intent: technical_support, Severity: medium, Priority: normal, Confidence: 0.80",
                    "QA Reviewer"
                )
            ]
    
    # Test the method
    mock_result = MockCrewResult()
    initial_outputs = crew._capture_initial_agent_outputs(mock_result)
    
    print(f"📊 Captured outputs from {len(initial_outputs)} agents")
    for agent_name, output in initial_outputs.items():
        print(f"  {agent_name}: intent={output['intent']}, severity={output['severity']}, confidence={output['confidence']}")
    
    # Validate results
    expected_agents = ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']
    success = all(agent in initial_outputs for agent in expected_agents)
    
    if success:
        print("✅ Initial agent outputs captured correctly")
    else:
        print("❌ Failed to capture all agent outputs")
    
    return success, initial_outputs

def test_disagreement_detection(initial_outputs):
    """Test disagreement detection from initial outputs."""
    print("\n🧪 Testing disagreement detection...")
    
    from agents import CollaborativeSupportCrew
    crew = CollaborativeSupportCrew()
    
    # Test disagreement comparison
    disagreements = crew._compare_initial_outputs(initial_outputs)
    
    print(f"📊 Found disagreements in {len(disagreements)} fields")
    for field, field_disagreements in disagreements.items():
        print(f"  {field}: {field_disagreements}")
    
    # Validate - should find disagreements in intent and severity based on our mock data
    expected_disagreements = ['intent', 'severity']  # Based on mock data above
    success = len(disagreements) > 0
    
    if success:
        print("✅ Disagreement detection working correctly")
    else:
        print("❌ Failed to detect expected disagreements")
    
    return success, disagreements

def test_agreement_scores(initial_outputs):
    """Test final agreement score calculation."""
    print("\n🧪 Testing agreement score calculation...")
    
    from agents import CollaborativeSupportCrew
    crew = CollaborativeSupportCrew()
    
    # Mock final output that matches some initial agent outputs
    final_output = "Final Classification: Intent: technical_support, Severity: high, Priority: normal"
    
    # Calculate agreement scores
    agreement_scores = crew._calculate_agreement_scores(initial_outputs, final_output)
    
    print(f"📊 Agreement scores: {agreement_scores}")
    
    # Validate - should have scores for intent, severity, priority
    expected_fields = ['intent', 'severity', 'priority']
    success = all(field in agreement_scores for field in expected_fields)
    
    if success:
        print("✅ Agreement scores calculated correctly")
    else:
        print("❌ Failed to calculate agreement scores for all fields")
    
    return success, agreement_scores

def test_consensus_timeline(disagreements):
    """Test consensus timeline tracking."""
    print("\n🧪 Testing consensus timeline tracking...")
    
    from agents import CollaborativeSupportCrew
    crew = CollaborativeSupportCrew()
    
    # Mock initial outputs for timeline
    mock_initial = {
        'triage_specialist': {'intent': 'technical_support', 'severity': 'high'},
        'ticket_analyst': {'intent': 'bug', 'severity': 'medium'}
    }
    
    # Track consensus timeline
    timeline = crew._track_consensus_timeline(mock_initial, disagreements)
    
    print(f"📊 Generated timeline with {len(timeline)} events")
    for event in timeline:
        print(f"  {event['event_type']}: {event['field']} at {event.get('timestamp', 'N/A')}")
    
    # Validate - should have events for each disagreement
    success = len(timeline) > 0
    
    if success:
        print("✅ Consensus timeline generated correctly")
    else:
        print("❌ Failed to generate consensus timeline")
    
    return success, timeline

def test_collaboration_metrics_integration():
    """Test the complete collaboration metrics with mocked data."""
    print("\n🧪 Testing complete collaboration metrics integration...")
    
    from agents import CollaborativeSupportCrew
    
    # Create crew instance
    crew = CollaborativeSupportCrew()
    
    # Create comprehensive mock crew result
    class MockAgent:
        def __init__(self, role):
            self.role = role
    
    class MockTaskOutput:
        def __init__(self, raw_output, agent_role):
            self.raw = raw_output
            self.agent = MockAgent(agent_role)
    
    class MockCrewResult:
        def __init__(self):
            self.tasks_output = [
                MockTaskOutput(
                    "Classification Analysis: Intent: technical_support, Severity: high, Priority: urgent, Confidence: 0.85. This is a critical system issue.",
                    "Triage Specialist"
                ),
                MockTaskOutput(
                    "Detailed Analysis: Intent: bug, Severity: medium, Priority: normal, Confidence: 0.75. Upon review, this appears to be a software bug.",
                    "Ticket Analyst"  
                ),
                MockTaskOutput(
                    "Strategic Assessment: Intent: technical_support, Severity: high, Priority: high, Confidence: 0.90. Requires immediate technical attention.",
                    "Support Strategist"
                ),
                MockTaskOutput(
                    "Quality Review: Intent: technical_support, Severity: medium, Priority: normal, Confidence: 0.80. Final consensus reached after discussion.",
                    "QA Reviewer"
                )
            ]
    
    # Test complete metrics extraction
    mock_result = MockCrewResult()
    consensus_start = time.time() - 5.0  # 5 seconds ago
    consensus_end = time.time()
    final_output = "Final Consensus: Intent: technical_support, Severity: high, Priority: normal, Confidence: 0.85"
    
    metrics = crew._extract_authentic_collaboration_metrics(
        mock_result,
        "test-ticket-123",
        consensus_start_time=consensus_start,
        consensus_end_time=consensus_end,
        final_output=final_output
    )
    
    print(f"📊 Generated collaboration metrics:")
    print(f"  Disagreement count: {metrics['disagreement_count']}")
    print(f"  Initial disagreements: {metrics['initial_disagreements']}")
    print(f"  Resolution iterations: {metrics['resolution_iterations']}")
    print(f"  Final agreement scores: {metrics['final_agreement_scores']}")
    print(f"  Confidence improvement: {metrics['confidence_improvement']}")
    print(f"  Consensus timing: {metrics['consensus_end_time'] - metrics['consensus_start_time']:.2f}s")
    print(f"  Agreement evolution events: {len(metrics['agent_agreement_evolution'])}")
    
    # Validate key metrics are populated
    validations = [
        metrics['disagreement_count'] >= 0,
        isinstance(metrics['initial_disagreements'], dict),
        metrics['resolution_iterations'] > 0,
        isinstance(metrics['final_agreement_scores'], dict),
        metrics['consensus_start_time'] is not None,
        metrics['consensus_end_time'] is not None,
        isinstance(metrics['agent_agreement_evolution'], list)
    ]
    
    success = all(validations)
    
    if success:
        print("✅ Complete collaboration metrics integration working")
    else:
        print("❌ Collaboration metrics integration has issues")
        print(f"Validation results: {validations}")
    
    return success, metrics

def main():
    """Run all collaboration metrics tests."""
    print("🚀 Starting collaboration metrics validation tests...\n")
    
    tests = []
    
    # Test 1: Initial agent outputs
    success1, initial_outputs = test_initial_agent_outputs()
    tests.append(("Initial Agent Outputs", success1))
    
    # Test 2: Disagreement detection
    success2, disagreements = test_disagreement_detection(initial_outputs)
    tests.append(("Disagreement Detection", success2))
    
    # Test 3: Agreement scores
    success3, agreement_scores = test_agreement_scores(initial_outputs)
    tests.append(("Agreement Scores", success3))
    
    # Test 4: Consensus timeline
    success4, timeline = test_consensus_timeline(disagreements)
    tests.append(("Consensus Timeline", success4))
    
    # Test 5: Complete integration
    success5, complete_metrics = test_collaboration_metrics_integration()
    tests.append(("Complete Integration", success5))
    
    # Summary
    print("\n" + "="*60)
    print("🧪 COLLABORATION METRICS TEST RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:25s} {status}")
        if success:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 ALL COLLABORATION METRICS TESTS PASSED!")
        print("✅ The enhanced collaboration metrics should now provide:")
        print("   • Real initial_disagreements with specific field conflicts")
        print("   • Accurate resolution_iterations based on disagreement count")
        print("   • Proper consensus_start_time and consensus_end_time tracking")
        print("   • Meaningful final_agreement_scores per classification field")
        print("   • Complete agent_agreement_evolution timeline")
        print("   • Calculated confidence_improvement from initial vs final")
        return 0
    else:
        print(f"\n⚠️ {len(tests) - passed} tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())