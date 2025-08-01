"""
Main application runner for the collaborative support ticket summarizer.
Demonstrates CrewAI collaborative multi-agent processing.
"""

import os
import sys
from typing import List, Dict, Any
from datetime import datetime

from config import setup_langsmith, setup_kaggle, DEFAULT_TICKET_LIMIT
from utils import (
    load_ticket_data, save_results, print_ticket_summary, 
    create_progress_bar, validate_environment
)
from agents import CollaborativeSupportCrew

def print_collaborative_summary(result: Dict[str, Any]):
    """Print a comprehensive summary of collaborative processing results."""
    print(f"\n{'='*80}")
    print(f"🎫 COLLABORATIVE ANALYSIS COMPLETE: TICKET {result.get('ticket_id', 'N/A')}")
    print(f"{'='*80}")
    
    # Original message
    original = result.get('original_message', '')
    print(f"📝 Original Message: {original[:100]}{'...' if len(original) > 100 else ''}")
    
    # Classification results
    classification = result.get('classification', {})
    print(f"🏷️  Collaborative Classification:")
    print(f"   Intent: {classification.get('intent', 'N/A')}")
    print(f"   Severity: {classification.get('severity', 'N/A')}")
    print(f"   Confidence: {classification.get('confidence', 0):.2f}")
    print(f"   Reasoning: {classification.get('reasoning', 'N/A')}")
    
    # Summary
    summary = result.get('summary', 'N/A')
    print(f"📋 Collaborative Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}")
    
    # Action recommendations
    actions = result.get('action_recommendation', {})
    print(f"🎯 Collaborative Action Plan:")
    print(f"   Primary: {actions.get('primary_action', 'N/A')}")
    print(f"   Priority: {actions.get('priority', 'N/A')}")
    print(f"   Est. Resolution: {actions.get('estimated_resolution_time', 'N/A')}")
    if actions.get('secondary_actions'):
        print(f"   Secondary: {', '.join(actions.get('secondary_actions', []))}")
    if actions.get('notes'):
        notes = actions.get('notes', '')
        print(f"   Notes: {notes[:150]}{'...' if len(notes) > 150 else ''}")
    
    # Collaboration metrics
    metrics = result.get('collaboration_metrics', {})
    print(f"🤝 Collaboration Metrics:")
    print(f"   Agent Agreement: {metrics.get('agent_agreement', 0):.2f}")
    print(f"   Conflicts Resolved: {metrics.get('conflicts_resolved', 0)}")
    print(f"   Consensus Reached: {'✅' if metrics.get('consensus_reached', False) else '❌'}")
    
    print(f"{'='*80}")

def print_final_collaborative_summary(results: List[Dict[str, Any]]):
    """Print final summary of all collaborative processing results."""
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"📊 FINAL COLLABORATIVE PROCESSING SUMMARY")
    print(f"{'='*80}")
    
    # Basic stats
    total_tickets = len(results)
    successful = len([r for r in results if r.get('processing_status') == 'completed'])
    errors = total_tickets - successful
    
    print(f"📈 Total tickets processed: {total_tickets}")
    print(f"✅ Successfully processed: {successful}")
    print(f"❌ Processing errors: {errors}")
    
    # Intent and severity distribution
    intents = {}
    severities = {}
    agreement_scores = []
    
    for result in results:
        classification = result.get('classification', {})
        intent = classification.get('intent', 'unknown')
        severity = classification.get('severity', 'unknown')
        
        intents[intent] = intents.get(intent, 0) + 1
        severities[severity] = severities.get(severity, 0) + 1
        
        metrics = result.get('collaboration_metrics', {})
        if metrics.get('agent_agreement'):
            agreement_scores.append(metrics.get('agent_agreement', 0))
    
    # Print distributions
    print(f"🏷️  Intent Distribution:")
    for intent, count in sorted(intents.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_tickets) * 100
        print(f"   {intent}: {count} ({percentage:.1f}%)")
    
    print(f"🚨 Severity Distribution:")
    for severity, count in sorted(severities.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_tickets) * 100
        print(f"   {severity}: {count} ({percentage:.1f}%)")
    
    # Action recommendations
    actions = {}
    for result in results:
        action_rec = result.get('action_recommendation', {})
        primary_action = action_rec.get('primary_action', 'unknown')
        actions[primary_action] = actions.get(primary_action, 0) + 1
    
    print(f"🎯 Top Collaborative Actions:")
    for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True)[:3]:
        percentage = (count / total_tickets) * 100
        print(f"   {action}: {count} ({percentage:.1f}%)")
    
    # Collaboration effectiveness
    if agreement_scores:
        avg_agreement = sum(agreement_scores) / len(agreement_scores)
        print(f"🤝 Average Agent Agreement: {avg_agreement:.2f}")
    
    # Total conflicts resolved
    total_conflicts = sum(r.get('collaboration_metrics', {}).get('conflicts_resolved', 0) for r in results)
    consensus_count = sum(1 for r in results if r.get('collaboration_metrics', {}).get('consensus_reached', False))
    
    print(f"🔧 Total Conflicts Resolved: {total_conflicts}")
    print(f"✅ Consensus Achieved: {consensus_count}/{total_tickets}")
    
    print(f"🔍 LangSmith Project: ticket-sum")
    print(f"📊 View detailed traces at: https://smith.langchain.com/")
    print(f"{'='*80}")

def main():
    """Main application entry point."""
    print("🎫 Support Ticket Summarizer - CrewAI Multi-Agent System")
    print("=" * 60)
    
    # Validate environment
    if not validate_environment():
        print("❌ Environment validation failed. Please check your .env file.")
        sys.exit(1)
    
    # Setup LangSmith tracing
    setup_langsmith()
    
    # Setup Kaggle credentials
    setup_kaggle()
    
    try:
        # Load ticket data with configurable limit
        print(f"\n📊 Loading ticket data (limit: {DEFAULT_TICKET_LIMIT})...")
        df = load_ticket_data(max_tickets=DEFAULT_TICKET_LIMIT)
        
        if df.empty:
            print("❌ No ticket data available. Exiting.")
            sys.exit(1)
        
        print(f"✅ Loaded {len(df)} tickets for processing")
        df_demo = df  # Use all loaded tickets since we already limited them
        
        # Initialize collaborative crew
        print("\n🤖 Initializing CrewAI Collaborative Support Crew...")
        print("   - Triage Specialist (Classification)")
        print("   - Ticket Analyst (Analysis & Summary)")
        print("   - Support Strategist (Action Planning)")
        print("   - QA Reviewer (Consensus & Validation)")
        
        crew = CollaborativeSupportCrew()
        print("✅ All collaborative agents initialized successfully")
        
        # Process tickets collaboratively
        print(f"\n🤝 Processing {len(df_demo)} tickets using collaborative workflow...")
        results = []
        
        # Create progress bar
        progress_bar = create_progress_bar(len(df_demo), "Processing tickets")
        
        for index, row in df_demo.iterrows():
            try:
                ticket_id = str(row['ticket_id'])
                ticket_content = str(row['message'])
                
                print(f"\n{'='*80}")
                print(f"🎫 STARTING COLLABORATIVE PROCESSING: TICKET {ticket_id}")
                print(f"{'='*80}")
                
                # Process ticket through collaborative crew
                result = crew.process_ticket_collaboratively(ticket_id, ticket_content)
                results.append(result)
                
                # Print collaborative summary to terminal
                print_collaborative_summary(result)
                
                # Update progress
                progress_bar.update(1)
                
            except Exception as e:
                print(f"❌ Error processing ticket {row.get('ticket_id', index)}: {str(e)}")
                # Add error result
                error_result = {
                    "ticket_id": str(row.get('ticket_id', index)),
                    "original_message": str(row.get('message', '')),
                    "classification": {"intent": "error", "severity": "unknown", "confidence": 0.0, "reasoning": str(e)},
                    "summary": f"Error processing ticket: {str(e)}",
                    "action_recommendation": {"primary_action": "manual_review", "priority": "high", "notes": str(e)},
                    "processing_status": "error"
                }
                results.append(error_result)
                progress_bar.update(1)
                continue
        
        progress_bar.close()
        
        # Save results
        print(f"\n💾 Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"results_{timestamp}.json"
        save_results(results, results_filename)
        
        # Print final collaborative summary
        print_final_collaborative_summary(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)

def print_final_summary(results: List[Dict[str, Any]]) -> None:
    """Print a final summary of all processed tickets."""
    print("\n" + "="*80)
    print("📊 FINAL PROCESSING SUMMARY")
    print("="*80)
    
    total_tickets = len(results)
    successful_tickets = len([r for r in results if r.get('processing_status') == 'completed'])
    error_tickets = total_tickets - successful_tickets
    
    print(f"📈 Total tickets processed: {total_tickets}")
    print(f"✅ Successfully processed: {successful_tickets}")
    print(f"❌ Processing errors: {error_tickets}")
    
    if successful_tickets > 0:
        # Intent distribution
        intents = {}
        severities = {}
        actions = {}
        
        for result in results:
            if result.get('processing_status') == 'completed':
                classification = result.get('classification', {})
                action = result.get('action_recommendation', {})
                
                intent = classification.get('intent', 'unknown')
                severity = classification.get('severity', 'unknown')
                primary_action = action.get('primary_action', 'unknown')
                
                intents[intent] = intents.get(intent, 0) + 1
                severities[severity] = severities.get(severity, 0) + 1
                actions[primary_action] = actions.get(primary_action, 0) + 1
        
        print(f"\n🏷️  Intent Distribution:")
        for intent, count in sorted(intents.items()):
            percentage = (count / successful_tickets) * 100
            print(f"   {intent}: {count} ({percentage:.1f}%)")
        
        print(f"\n🚨 Severity Distribution:")
        for severity, count in sorted(severities.items()):
            percentage = (count / successful_tickets) * 100
            print(f"   {severity}: {count} ({percentage:.1f}%)")
        
        print(f"\n🎯 Top Actions:")
        for action, count in sorted(actions.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = (count / successful_tickets) * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")
    
    print(f"\n🔍 LangSmith Project: ticket-sum")
    print(f"📊 View detailed traces at: https://smith.langchain.com/")
    print("="*80)

if __name__ == "__main__":
    main()
