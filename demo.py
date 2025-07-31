"""
Demo script for the support ticket summarizer with sample data.
This script demonstrates the full functionality without requiring API calls.
"""

import json
from datetime import datetime
from utils import print_ticket_summary, save_results

def demo_support_ticket_summarizer():
    """Run a demonstration of the support ticket summarizer with sample data."""
    
    print("🎫 Support Ticket Summarizer - Demo Mode")
    print("=" * 60)
    print("🎯 Demonstrating multi-agent AI system with sample results")
    print()
    
    # Sample ticket data with realistic AI-generated results
    sample_tickets = [
        {
            "ticket_id": "DEMO001",
            "original_message": "My account has been charged twice for the same transaction. I need a refund immediately.",
            "classification": {
                "intent": "billing",
                "severity": "high", 
                "confidence": 0.95,
                "reasoning": "Customer reporting duplicate billing charge requiring immediate refund"
            },
            "summary": "Customer reports duplicate billing for same transaction and requests immediate refund. High priority billing issue requiring urgent financial review.",
            "action_recommendation": {
                "primary_action": "route_to_billing",
                "secondary_actions": ["immediate_response", "escalate_to_manager"],
                "priority": "high",
                "estimated_resolution_time": "2-4 hours",
                "notes": "Verify duplicate charge in billing system and process refund if confirmed"
            },
            "processing_status": "completed"
        },
        {
            "ticket_id": "DEMO002", 
            "original_message": "The mobile app keeps crashing when I try to upload photos. This is very frustrating.",
            "classification": {
                "intent": "bug",
                "severity": "medium",
                "confidence": 0.92,
                "reasoning": "Technical issue with app functionality causing user frustration"
            },
            "summary": "User experiencing app crashes during photo upload functionality. Technical bug affecting core app features.",
            "action_recommendation": {
                "primary_action": "route_to_technical",
                "secondary_actions": ["request_more_info"],
                "priority": "medium", 
                "estimated_resolution_time": "4-8 hours",
                "notes": "Collect device info, app version, and reproduction steps for engineering team"
            },
            "processing_status": "completed"
        },
        {
            "ticket_id": "DEMO003",
            "original_message": "I love the new features you added! The user interface is much more intuitive now.",
            "classification": {
                "intent": "compliment",
                "severity": "low",
                "confidence": 0.98,
                "reasoning": "Positive feedback about product improvements"
            },
            "summary": "Customer expressing satisfaction with recent UI improvements and new features. Positive feedback on product experience.",
            "action_recommendation": {
                "primary_action": "respond_with_template",
                "secondary_actions": [],
                "priority": "low",
                "estimated_resolution_time": "1 hour",
                "notes": "Send thank you response and forward feedback to product team"
            },
            "processing_status": "completed"
        },
        {
            "ticket_id": "DEMO004",
            "original_message": "How do I change my password? I can't find the option in settings.",
            "classification": {
                "intent": "general_inquiry", 
                "severity": "low",
                "confidence": 0.89,
                "reasoning": "Customer needs help with basic account functionality"
            },
            "summary": "User needs assistance locating password change functionality in account settings. Simple help request.",
            "action_recommendation": {
                "primary_action": "respond_with_template",
                "secondary_actions": [],
                "priority": "low",
                "estimated_resolution_time": "30 minutes",
                "notes": "Provide step-by-step password change instructions"
            },
            "processing_status": "completed"
        },
        {
            "ticket_id": "DEMO005",
            "original_message": "Critical security vulnerability found in your API endpoint. Please contact me ASAP.",
            "classification": {
                "intent": "technical_support",
                "severity": "critical",
                "confidence": 0.96,
                "reasoning": "Security vulnerability report requiring immediate attention"
            },
            "summary": "User reporting critical security vulnerability in API endpoint. Requires immediate security team review and response.",
            "action_recommendation": {
                "primary_action": "escalate_to_manager",
                "secondary_actions": ["immediate_response", "escalate_to_tier2"],
                "priority": "critical",
                "estimated_resolution_time": "1 hour",
                "notes": "Immediately escalate to security team and acknowledge receipt within 15 minutes"
            },
            "processing_status": "completed"
        }
    ]
    
    print("🔄 Processing 5 demonstration tickets...")
    print()
    
    # Display each ticket summary
    for i, ticket in enumerate(sample_tickets, 1):
        print(f"Processing ticket {i}/5...")
        print_ticket_summary(ticket)
        print()
    
    # Save demo results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_filename = f"demo_results_{timestamp}.json"
    save_results(sample_tickets, demo_filename)
    
    # Print final summary
    print_final_demo_summary(sample_tickets)

def print_final_demo_summary(results):
    """Print a summary of the demo results."""
    print("\n" + "="*80)
    print("📊 DEMO PROCESSING SUMMARY")
    print("="*80)
    
    # Intent distribution
    intents = {}
    severities = {}
    actions = {}
    
    for result in results:
        classification = result.get('classification', {})
        action = result.get('action_recommendation', {})
        
        intent = classification.get('intent', 'unknown')
        severity = classification.get('severity', 'unknown')
        primary_action = action.get('primary_action', 'unknown')
        
        intents[intent] = intents.get(intent, 0) + 1
        severities[severity] = severities.get(severity, 0) + 1
        actions[primary_action] = actions.get(primary_action, 0) + 1
    
    print(f"📈 Total tickets processed: {len(results)}")
    print(f"✅ Successfully processed: {len(results)}")
    print(f"❌ Processing errors: 0")
    
    print(f"\n🏷️  Intent Distribution:")
    for intent, count in sorted(intents.items()):
        percentage = (count / len(results)) * 100
        print(f"   {intent}: {count} ({percentage:.1f}%)")
    
    print(f"\n🚨 Severity Distribution:")
    for severity, count in sorted(severities.items()):
        percentage = (count / len(results)) * 100
        print(f"   {severity}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Action Distribution:")
    for action, count in sorted(actions.items()):
        percentage = (count / len(results)) * 100
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    print(f"\n✨ This demonstrates the full CrewAI multi-agent workflow:")
    print(f"   🏷️  ClassifierAgent: Identifies intent and severity")
    print(f"   📋 SummarizerAgent: Creates concise ticket summaries") 
    print(f"   🎯 ActionRecommenderAgent: Suggests next steps")
    print("="*80)

if __name__ == "__main__":
    demo_support_ticket_summarizer()