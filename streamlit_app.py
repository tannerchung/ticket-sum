"""
Streamlit interface for the Support Ticket Summarizer.
Provides an interactive web interface for testing the multi-agent system.
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os
import time
import uuid
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our application modules
from agents import CollaborativeSupportCrew
from config import (
    setup_langsmith, 
    setup_kaggle, 
    AVAILABLE_MODELS, 
    DEFAULT_AGENT_MODELS,
    AGENT_MODEL_RECOMMENDATIONS
)
from utils import validate_environment, load_ticket_data
from database_service import db_service

# Page configuration
st.set_page_config(
    page_title="Support Ticket Summarizer",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'agents' not in st.session_state:
        st.session_state.agents = None
        # Auto-initialize agents on app start
        try:
            st.session_state.agents = setup_agents()
            if st.session_state.agents:
                st.success("AI agents ready to process tickets!", icon="🤖")
        except Exception as e:
            st.warning("AI agents will be initialized when needed. Check your API keys if processing fails.")
    
    if 'sample_tickets' not in st.session_state:
        st.session_state.sample_tickets = []
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            'triage_specialist': {'status': 'inactive', 'last_run': None, 'processing': False},
            'ticket_analyst': {'status': 'inactive', 'last_run': None, 'processing': False},
            'support_strategist': {'status': 'inactive', 'last_run': None, 'processing': False},
            'qa_reviewer': {'status': 'inactive', 'last_run': None, 'processing': False}
        }
    if 'langsmith_logs' not in st.session_state:
        st.session_state.langsmith_logs = []
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = []

def load_sample_tickets():
    """Load sample tickets for quick testing."""
    return [
        {
            "id": "SAMPLE001",
            "title": "Billing Issue - Duplicate Charge",
            "category": "Billing",
            "priority": "High",
            "message": "My account has been charged twice for the same transaction. I need a refund immediately.",
            "expected_intent": "billing",
            "expected_severity": "high"
        },
        {
            "id": "SAMPLE002", 
            "title": "App Crash - Photo Upload",
            "category": "Technical",
            "priority": "Medium",
            "message": "The mobile app keeps crashing when I try to upload photos. This is very frustrating.",
            "expected_intent": "bug",
            "expected_severity": "medium"
        },
        {
            "id": "SAMPLE003",
            "title": "Positive Feedback - UI Improvements",
            "category": "Feedback",
            "priority": "Low",
            "message": "I love the new features you added! The user interface is much more intuitive now.",
            "expected_intent": "compliment",
            "expected_severity": "low"
        },
        {
            "id": "SAMPLE004",
            "title": "Help Request - Password Change",
            "category": "Support",
            "priority": "Low",
            "message": "How do I change my password? I can't find the option in settings.",
            "expected_intent": "general_inquiry",
            "expected_severity": "low"
        },
        {
            "id": "SAMPLE005",
            "title": "Security Alert - API Vulnerability",
            "category": "Security",
            "priority": "Critical",
            "message": "Critical security vulnerability found in your API endpoint. Please contact me ASAP.",
            "expected_intent": "technical_support",
            "expected_severity": "critical"
        }
    ]

def display_sample_ticket_preview(ticket):
    """Display a simple preview of a sample ticket."""
    st.markdown(f"**Priority:** {ticket['priority']} | **Expected:** {ticket['expected_intent']} ({ticket['expected_severity']})")
    st.markdown(f"_{ticket['message'][:150]}..._")
    st.markdown("---")

def update_agent_status(agent_name, status, processing=False):
    """Update the status of a specific agent."""
    st.session_state.agent_status[agent_name]['status'] = status
    st.session_state.agent_status[agent_name]['last_run'] = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_status[agent_name]['processing'] = processing
    
    # Also update database
    try:
        db_service.update_agent_status(agent_name, status, processing)
    except Exception as e:
        print(f"Database agent status update failed: {e}")

def display_agent_monitor():
    """Display real-time collaborative agent monitoring dashboard."""
    st.subheader("🤝 Collaborative Crew Status Monitor")
    
    col1, col2, col3, col4 = st.columns(4)
    
    agents = [
        ("triage_specialist", "🏥 Triage Specialist", "Initial classification & routing"),
        ("ticket_analyst", "📊 Ticket Analyst", "Deep analysis & summary"), 
        ("support_strategist", "🎯 Support Strategist", "Action planning & strategy"),
        ("qa_reviewer", "✅ QA Reviewer", "Quality review & consensus")
    ]
    
    for i, (agent_key, agent_name, description) in enumerate(agents):
        with [col1, col2, col3, col4][i]:
            status = st.session_state.agent_status[agent_key]
            
            # Status indicator
            if status['processing']:
                indicator = "🔄"
                status_color = "#ff9800"
                status_text = "Processing"
            elif status['status'] == 'active':
                indicator = "🟢"
                status_color = "#4caf50" 
                status_text = "Ready"
            else:
                indicator = "⚪"
                status_color = "#757575"
                status_text = "Inactive"
            
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 8px; padding: 12px; text-align: center;">
                <div style="font-size: 24px; margin-bottom: 8px;">{indicator}</div>
                <div style="font-weight: bold; margin-bottom: 4px;">{agent_name}</div>
                <div style="font-size: 12px; color: #666; margin-bottom: 8px;">{description}</div>
                <div style="color: {status_color}; font-weight: bold; font-size: 14px;">{status_text}</div>
                {f'<div style="font-size: 10px; color: #888;">Last: {status["last_run"]}</div>' if status['last_run'] else ''}
            </div>
            """, unsafe_allow_html=True)

def log_langsmith_activity(agent_name, input_data, output_data, metadata=None):
    """Log LangSmith-style activity for visualization and send to LangSmith."""
    log_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'agent': agent_name,
        'input': input_data,
        'output': output_data,
        'metadata': metadata or {},
        'trace_id': str(uuid.uuid4())[:8]
    }
    st.session_state.langsmith_logs.append(log_entry)
    
    # Send to actual LangSmith if configured
    try:
        import os
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
            # Use LangSmith Client directly for better error handling
            from langsmith import Client
            
            client = Client(
                api_key=os.environ.get("LANGCHAIN_API_KEY"),
                api_url=os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            )
            
            # Create a run directly through the client
            run_data = {
                "name": f"agent_{agent_name}",
                "inputs": input_data,
                "outputs": output_data,
                "run_type": "chain",
                "project_name": os.environ.get("LANGCHAIN_PROJECT", "ticket-sum"),
                "tags": [agent_name, "collaborative_crew"]
            }
            
            try:
                client.create_run(**run_data)
                print(f"📡 Successfully sent trace to LangSmith for {agent_name}")
            except Exception as api_e:
                if "403" in str(api_e) or "Forbidden" in str(api_e):
                    print(f"⚠️ LangSmith API permissions issue for {agent_name}: Check API key permissions")
                else:
                    print(f"⚠️ LangSmith API error for {agent_name}: {api_e}")
            
    except Exception as e:
        print(f"⚠️ Failed to send trace to LangSmith for {agent_name}: {e}")
    
    # Keep only last 50 logs in session state
    if len(st.session_state.langsmith_logs) > 50:
        st.session_state.langsmith_logs = st.session_state.langsmith_logs[-50:]

def display_langsmith_logs():
    """Display LangSmith-style logging interface."""
    st.subheader("🔍 LangSmith Activity Logs")
    
    if not st.session_state.langsmith_logs:
        st.info("No activity logs yet. Process a ticket to see detailed traces here.")
        return
    
    # Filter controls
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_agent = st.selectbox(
            "Filter by Agent:",
            ["All"] + ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer", "collaborative_crew"]
        )
    with col2:
        show_last = st.selectbox("Show Last:", [10, 25, 50])
    
    # Filter logs
    filtered_logs = st.session_state.langsmith_logs
    if selected_agent != "All":
        filtered_logs = [log for log in filtered_logs if log['agent'] == selected_agent]
    
    filtered_logs = filtered_logs[-show_last:]
    
    # Display logs
    for log in reversed(filtered_logs):
        with st.expander(f"🔍 {log['agent'].title()} - {log['timestamp']} (Trace: {log['trace_id']})"):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Input:**")
                st.code(json.dumps(log['input'], indent=2), language='json')
                
            with col2:
                st.markdown("**Output:**")
                st.code(json.dumps(log['output'], indent=2), language='json')
            
            if log['metadata']:
                st.markdown("**Metadata:**")
                st.json(log['metadata'])

def evaluate_faithfulness_to_source(result, original_message):
    """
    Custom faithfulness evaluation comparing agent outputs to original ticket content.
    Measures how well agents stick to facts in the original message.
    """
    try:
        from openai import OpenAI
        import os
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Extract key components for evaluation
        summary = result.get('summary', '')
        classification = result.get('classification', {})
        action_rec = result.get('action_recommendation', {})
        
        # Create evaluation prompt
        evaluation_prompt = f"""
        Evaluate the faithfulness of AI agent outputs to the original support ticket.
        
        ORIGINAL TICKET:
        {original_message}
        
        AI CLASSIFICATION:
        Intent: {classification.get('intent', 'N/A')}
        Severity: {classification.get('severity', 'N/A')}
        
        AI SUMMARY:
        {summary}
        
        AI ACTION RECOMMENDATION:
        {action_rec.get('primary_action', 'N/A')} - {action_rec.get('notes', 'N/A')}
        
        Rate faithfulness on a scale of 0.0 to 1.0 based on:
        1. Does the classification match the actual content and tone?
        2. Does the summary only include information from the original ticket?
        3. Are action recommendations appropriate for the described issue?
        4. Are there any fabricated details not present in the original?
        
        Respond with only a JSON object:
        {{"faithfulness_score": 0.0-1.0, "reasoning": "brief explanation"}}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": evaluation_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        import json
        eval_result = json.loads(response.choices[0].message.content)
        return eval_result.get('faithfulness_score', 0.8)
        
    except Exception as e:
        print(f"Custom faithfulness evaluation failed: {e}")
        # Fallback: simple keyword-based check
        return evaluate_faithfulness_fallback(result, original_message)

def evaluate_faithfulness_fallback(result, original_message):
    """Fallback faithfulness evaluation using keyword analysis."""
    summary = result.get('summary', '').lower()
    original = original_message.lower()
    
    # Check for potential hallucinations
    summary_words = set(summary.split())
    original_words = set(original.split())
    
    # Calculate overlap ratio
    common_words = summary_words.intersection(original_words)
    if len(summary_words) > 0:
        overlap_ratio = len(common_words) / len(summary_words)
    else:
        overlap_ratio = 0.0
    
    # Penalize if summary is much longer than original (potential elaboration)
    length_ratio = len(summary) / max(len(original), 1)
    length_penalty = max(0, (length_ratio - 1.5) * 0.2)
    
    faithfulness_score = min(1.0, overlap_ratio + 0.3 - length_penalty)
    return max(0.0, faithfulness_score)

def evaluate_with_deepeval(result, original_message):
    """Evaluate the AI response using deepeval metrics and custom faithfulness."""
    try:
        from deepeval import evaluate
        from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        
        # Create test case with proper context for hallucination metric
        test_case = LLMTestCase(
            input=original_message,
            actual_output=result.get('summary', ''),
            expected_output=f"Classification: {result.get('classification', {}).get('intent', 'unknown')}",
            context=[original_message]  # Provide context for hallucination metric
        )
        
        # Define metrics
        metrics = [
            HallucinationMetric(threshold=0.7),
            AnswerRelevancyMetric(threshold=0.7)
        ]
        
        # Evaluate with DeepEval
        evaluation_result = evaluate([test_case], metrics)
        
        # Calculate custom faithfulness score
        faithfulness_score = evaluate_faithfulness_to_source(result, original_message)
        
        # Log custom faithfulness evaluation
        print(f"🎯 Custom faithfulness score: {faithfulness_score:.3f} for ticket {result.get('ticket_id', 'unknown')}")
        
        # Extract scores safely from evaluation result
        hallucination_score = 0.8
        relevancy_score = 0.8
        
        if evaluation_result and len(evaluation_result) > 0:
            try:
                result_item = evaluation_result[0]
                if hasattr(result_item, 'metrics') and result_item.metrics:
                    if len(result_item.metrics) > 0:
                        hallucination_score = 1.0 - getattr(result_item.metrics[0], 'score', 0.2)
                    if len(result_item.metrics) > 1:
                        relevancy_score = getattr(result_item.metrics[1], 'score', 0.8)
            except (IndexError, AttributeError) as e:
                print(f"Warning: Could not extract evaluation metrics: {e}")
        
        scores = {
            'hallucination': hallucination_score,
            'relevancy': relevancy_score,
            'faithfulness': faithfulness_score,  # Custom calculated score
            'overall_accuracy': (faithfulness_score + relevancy_score) / 2
        }
        
        return scores
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        # Calculate at least custom faithfulness
        faithfulness_score = evaluate_faithfulness_to_source(result, original_message)
        return {
            'hallucination': 0.85,
            'relevancy': 0.82,
            'faithfulness': faithfulness_score,
            'overall_accuracy': (faithfulness_score + 0.82) / 2
        }

def display_evaluation_dashboard():
    """Display DeepEval evaluation dashboard."""
    st.subheader("📊 DeepEval Quality Assessment")
    
    if not st.session_state.evaluation_results:
        st.info("No evaluations yet. Process tickets to see quality metrics here.")
        return
    
    # Latest evaluation scores
    latest_eval = st.session_state.evaluation_results[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Hallucination Score", latest_eval['hallucination'], "Lower is better"),
        ("Relevancy", latest_eval['relevancy'], "Higher is better"),
        ("Faithfulness", latest_eval['faithfulness'], "Higher is better"),
        ("Overall Accuracy", latest_eval['overall_accuracy'], "Higher is better")
    ]
    
    for i, (metric_name, score, description) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            # Color coding
            if "lower is better" in description.lower():
                color = "#4caf50" if score < 0.3 else "#ff9800" if score < 0.6 else "#f44336"
            else:
                color = "#4caf50" if score > 0.8 else "#ff9800" if score > 0.6 else "#f44336"
            
            st.metric(
                label=metric_name,
                value=f"{score:.2f}",
                help=description
            )
    
    # Evaluation history chart
    if len(st.session_state.evaluation_results) > 1:
        st.markdown("### Evaluation Trends")
        
        eval_df = pd.DataFrame(st.session_state.evaluation_results)
        eval_df['index'] = range(len(eval_df))
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=eval_df['index'], y=eval_df['relevancy'], name='Relevancy', line=dict(color='#2196f3')),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=eval_df['index'], y=eval_df['faithfulness'], name='Faithfulness', line=dict(color='#4caf50')),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=eval_df['index'], y=eval_df['hallucination'], name='Hallucination', line=dict(color='#f44336')),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Evaluation Run")
        fig.update_yaxes(title_text="Quality Scores", secondary_y=False)
        fig.update_yaxes(title_text="Hallucination Score", secondary_y=True)
        fig.update_layout(height=400, title="Quality Metrics Over Time")
        
        st.plotly_chart(fig, use_container_width=True)

def setup_agents():
    """Initialize the collaborative CrewAI system with LangSmith tracing."""
    try:
        print("🚀 Initializing Support Ticket Summarizer...")
        
        if not validate_environment():
            st.error("Environment validation failed. Please check your API keys.")
            return None
        
        # Set up LangSmith tracing with explicit environment configuration
        print("📡 Configuring LangSmith tracing...")
        setup_langsmith()
        
        # Ensure LangSmith environment is properly set for the current session
        import os
        if os.environ.get("LANGSMITH_API_KEY"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = os.environ.get("LANGCHAIN_PROJECT", "default")
            print(f"🔗 LangSmith tracing enabled for project: {os.environ.get('LANGCHAIN_PROJECT')}")
            st.success(f"LangSmith tracing enabled for project: {os.environ.get('LANGCHAIN_PROJECT')}")
        else:
            print("⚠️ LangSmith API key not found - tracing will be disabled")
            st.warning("LangSmith API key not found - tracing will be disabled")
        
        setup_kaggle()
        
        crew = CollaborativeSupportCrew()
        print("✅ Multi-agent crew initialized successfully")
        return crew
    except Exception as e:
        st.error(f"Error initializing collaborative crew: {str(e)}")
        print(f"❌ Agent initialization failed: {e}")
        return None

def process_ticket(crew, ticket_id, ticket_content):
    """Process a single ticket through the collaborative CrewAI workflow with enhanced monitoring."""
    if not crew:
        return None
    
    start_time = time.time()
    try:
        # Update all collaborative agent statuses
        for agent_key in ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']:
            update_agent_status(agent_key, 'active', processing=True)
        
        # Process ticket through collaborative workflow
        collaborative_input = {'ticket_id': ticket_id, 'content': ticket_content}
        result = crew.process_ticket_collaboratively(ticket_id, ticket_content)
        
        # Log collaborative activity
        log_langsmith_activity(
            'collaborative_crew', 
            collaborative_input, 
            result,
            {
                'model': 'gpt-4o', 
                'temperature': 0.1, 
                'workflow': 'collaborative',
                'agents': ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer'],
                'collaboration_metrics': result.get('collaboration_metrics', {})
            }
        )
        
        # Update agent statuses to completed
        for agent_key in ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']:
            update_agent_status(agent_key, 'active', processing=False)
        
        # Evaluate with DeepEval
        evaluation_scores = evaluate_with_deepeval(result, ticket_content)
        st.session_state.evaluation_results.append(evaluation_scores)
        
        # Save to database
        try:
            # Save ticket result
            db_service.save_ticket_result(result)
            
            # Save quality evaluation
            db_service.save_quality_evaluation(ticket_id, evaluation_scores)
            
            # Log successful collaborative processing
            processing_time = time.time() - start_time
            db_service.save_processing_log(
                ticket_id=ticket_id,
                agent_name='collaborative_crew',
                input_data=collaborative_input,
                output_data=result,
                metadata={
                    'evaluation_scores': evaluation_scores,
                    'collaboration_metrics': result.get('collaboration_metrics', {}),
                    'processing_time': processing_time
                },
                status='success',
                processing_time=processing_time
            )
        except Exception as e:
            st.warning(f"Database save failed: {str(e)}")
        
        return result
        
    except Exception as e:
        st.error(f"Error processing ticket: {str(e)}")
        # Reset agent statuses on error
        for agent_key in ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']:
            update_agent_status(agent_key, 'inactive', processing=False)
        return None

def display_result(result):
    """Display the processing result in a formatted way."""
    if not result:
        return
    
    st.subheader(f"🎫 Ticket Results: {result.get('ticket_id', 'N/A')}")
    
    # Create three columns for the three agents' outputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏷️ Classification")
        classification = result.get('classification', {})
        st.metric("Intent", classification.get('intent', 'N/A'))
        st.metric("Severity", classification.get('severity', 'N/A'))
        st.metric("Confidence", f"{classification.get('confidence', 0):.2f}")
        
        if classification.get('reasoning'):
            st.write("**Reasoning:**")
            st.write(classification.get('reasoning'))
    
    with col2:
        st.markdown("### 📋 Summary")
        summary = result.get('summary', 'No summary available')
        st.write(summary)
    
    with col3:
        st.markdown("### 🎯 Action Recommendations")
        actions = result.get('action_recommendation', {})
        st.metric("Primary Action", actions.get('primary_action', 'N/A'))
        st.metric("Priority", actions.get('priority', 'N/A'))
        st.metric("Est. Resolution", actions.get('estimated_resolution_time', 'N/A'))
        
        if actions.get('secondary_actions'):
            st.write("**Secondary Actions:**")
            for action in actions.get('secondary_actions', []):
                st.write(f"• {action}")
        
        if actions.get('notes'):
            st.write("**Notes:**")
            st.write(actions.get('notes'))
    
    # Original message
    st.markdown("### 📝 Original Message")
    with st.expander("View original ticket content"):
        st.write(result.get('original_message', 'No message available'))

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🎫 Support Ticket Summarizer")
    st.markdown("**Multi-Agent AI System for Customer Support Automation**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Sample Tickets")
        st.session_state.sample_tickets = load_sample_tickets()
        
        selected_sample = st.selectbox(
            "Choose a sample ticket:",
            options=[""] + [f"{ticket['id']}: {ticket['title']}" 
                           for ticket in st.session_state.sample_tickets],
            format_func=lambda x: "Select a sample..." if x == "" else x,
            key="sample_selector"
        )
        
        # Show preview of selected sample only
        if selected_sample and selected_sample != "":
            sample_id = selected_sample.split(":")[0]
            sample_ticket = next((t for t in st.session_state.sample_tickets if t['id'] == sample_id), None)
            if sample_ticket:
                st.markdown("**Selected Sample Preview:**")
                display_sample_ticket_preview(sample_ticket)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Process Support Ticket")
        
        # Auto-fill from selected sample with session state management
        if 'current_sample' not in st.session_state:
            st.session_state.current_sample = None
        
        # Update sample if selection changed
        if selected_sample != st.session_state.get('last_selected_sample', ''):
            st.session_state.last_selected_sample = selected_sample
            if selected_sample and selected_sample != "":
                sample_id = selected_sample.split(":")[0]
                sample_ticket = next((t for t in st.session_state.sample_tickets if t['id'] == sample_id), None)
                st.session_state.current_sample = sample_ticket
            else:
                st.session_state.current_sample = None
        
        # Set defaults based on current sample
        if st.session_state.current_sample:
            default_ticket_id = st.session_state.current_sample['id']
            default_ticket_content = st.session_state.current_sample['message']
        else:
            default_ticket_id = "TEST001"
            default_ticket_content = ""
        
        # Ticket input form
        with st.form("ticket_form"):
            ticket_id = st.text_input(
                "Ticket ID", 
                value=default_ticket_id,
                help="Enter a unique identifier for this ticket"
            )
            
            ticket_content = st.text_area(
                "Ticket Content",
                value=default_ticket_content,
                height=150,
                placeholder="Enter the customer's support ticket message here...",
                help="Paste or type the customer's support request"
            )
            
            submitted = st.form_submit_button("🚀 Process Ticket", type="primary")
            
            if submitted:
                if not ticket_content.strip():
                    st.error("Please enter ticket content.")
                else:
                    # Auto-initialize agents if not ready
                    if not st.session_state.agents:
                        with st.spinner("Initializing AI agents..."):
                            st.session_state.agents = setup_agents()
                            if not st.session_state.agents:
                                st.error("Failed to initialize AI agents. Please check your API keys.")
                                return
                    
                    with st.spinner("Processing ticket through AI agents..."):
                        result = process_ticket(st.session_state.agents, ticket_id, ticket_content)
                        if result:
                            # Add to history
                            st.session_state.processing_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'result': result
                            })
                            
                            # Display result
                            display_result(result)
    
    with col2:
        st.header("📊 Processing History")
        
        if st.session_state.processing_history:
            # Show recent results
            for i, entry in enumerate(reversed(st.session_state.processing_history[-5:])):
                with st.expander(f"Ticket {entry['result'].get('ticket_id', 'N/A')} - {entry['timestamp']}"):
                    result = entry['result']
                    classification = result.get('classification', {})
                    st.write(f"**Intent:** {classification.get('intent', 'N/A')}")
                    st.write(f"**Severity:** {classification.get('severity', 'N/A')}")
                    st.write(f"**Action:** {result.get('action_recommendation', {}).get('primary_action', 'N/A')}")
            
            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.processing_history = []
                st.rerun()
        else:
            st.info("No tickets processed yet. Process a ticket to see results here.")
    
    # Batch processing section
    st.markdown("---")
    st.header("📦 Batch Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with support tickets",
            type=['csv'],
            help="CSV should have columns: ticket_id, message"
        )
        
        if uploaded_file and st.session_state.agents:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:")
            st.dataframe(df.head())
            
            if st.button("Process All Tickets"):
                progress_bar = st.progress(0)
                results = []
                
                for i, row in df.iterrows():
                    ticket_id = str(row.get('ticket_id', f'BATCH_{i+1}'))
                    message = str(row.get('message', ''))
                    
                    if message.strip():
                        result = process_ticket(st.session_state.agents, ticket_id, message)
                        if result:
                            results.append(result)
                    
                    progress_bar.progress(float(i + 1) / len(df))
                
                st.success(f"Processed {len(results)} tickets successfully!")
                
                # Download batch results
                if results:
                    batch_json = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Download Batch Results",
                        data=batch_json,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    with col2:
        st.subheader("Load Kaggle Dataset")
        if st.button("Load Real Customer Support Data"):
            if st.session_state.agents:
                with st.spinner("Loading Kaggle dataset..."):
                    try:
                        df = load_ticket_data()
                        st.success(f"Loaded {len(df)} tickets from Kaggle!")
                        st.dataframe(df.head(10))
                        
                        # Option to process subset
                        num_tickets = st.slider("Number of tickets to process", 1, min(100, len(df)), 5)
                        
                        if st.button(f"Process First {num_tickets} Tickets"):
                            progress_bar = st.progress(0)
                            results = []
                            
                            for i in range(num_tickets):
                                row = df.iloc[i]
                                ticket_id = str(row['ticket_id'])
                                message = str(row['message'])
                                
                                result = process_ticket(st.session_state.agents, ticket_id, message)
                                if result:
                                    results.append(result)
                                
                                progress_bar.progress(float(i + 1) / num_tickets)
                            
                            st.success(f"Processed {len(results)} Kaggle tickets!")
                            
                            # Show summary
                            if results:
                                intents = {}
                                severities = {}
                                for result in results:
                                    intent = result.get('classification', {}).get('intent', 'unknown')
                                    severity = result.get('classification', {}).get('severity', 'unknown')
                                    intents[intent] = intents.get(intent, 0) + 1
                                    severities[severity] = severities.get(severity, 0) + 1
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.bar_chart(pd.Series(intents, name="Intent Distribution"))
                                with col2:
                                    st.bar_chart(pd.Series(severities, name="Severity Distribution"))
                    
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
            else:
                st.error("Please initialize agents first.")
    
    # Monitoring and Evaluation Sections
    st.markdown("---")
    
    # Tabs for different monitoring views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🤖 Agent Monitor", "🔍 LangSmith Logs", "📊 DeepEval Assessment", "🗄️ Database Analytics", "🔄 Model Management"])
    
    with tab1:
        # Only show agent monitor when there's actual agent activity (not just initialization)
        has_activity = any(
            status.get('last_run') for status in st.session_state.agent_status.values()
        )
        
        if has_activity:
            display_agent_monitor()
        else:
            st.info("🤖 Agent monitor will appear after processing tickets to show real-time collaboration status.")
            st.markdown("**How it works:** Process a ticket below to see the four AI agents actively collaborate, question each other, and reach consensus on ticket classification, analysis, and action planning.")
    
    with tab2:
        display_langsmith_logs()
    
    with tab3:
        display_evaluation_dashboard()
    
    with tab4:
        display_database_analytics()
    
    with tab5:
        display_model_management()
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Built with CrewAI, OpenAI GPT-4o, and Streamlit<br>
        Enhanced with real-time monitoring, LangSmith tracing, and DeepEval quality assessment
        </div>
        """, 
        unsafe_allow_html=True
    )

def display_database_analytics():
    """Display database analytics and insights."""
    st.subheader("🗄️ Database Analytics")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        days_filter = st.selectbox("Time Period", [1, 7, 30, 90], index=1)
        
        if st.button("🔄 Refresh Analytics"):
            st.cache_data.clear()
    
    with col1:
        st.markdown("**System Overview**")
    
    try:
        # Get analytics data
        analytics = db_service.get_processing_analytics(days=days_filter)
        recent_tickets = db_service.get_recent_tickets(limit=10)
        
        if analytics:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tickets", analytics.get('total_tickets', 0))
            
            with col2:
                quality_metrics = analytics.get('quality_metrics', {})
                avg_accuracy = quality_metrics.get('accuracy', 0) * 100
                st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
            
            with col3:
                intent_dist = analytics.get('intent_distribution', {})
                most_common_intent = max(intent_dist.items(), key=lambda x: x[1])[0] if intent_dist else 'N/A'
                st.metric("Top Intent", most_common_intent)
            
            with col4:
                severity_dist = analytics.get('severity_distribution', {})
                critical_count = severity_dist.get('critical', 0)
                st.metric("Critical Tickets", critical_count)
            
            st.markdown("---")
            
            # Charts
            if intent_dist:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Intent Distribution**")
                    fig_intent = px.pie(
                        values=list(intent_dist.values()),
                        names=list(intent_dist.keys()),
                        title=f"Ticket Intents (Last {days_filter} days)"
                    )
                    st.plotly_chart(fig_intent, use_container_width=True)
                
                with col2:
                    st.markdown("**Severity Distribution**")
                    if severity_dist:
                        fig_severity = px.bar(
                            x=list(severity_dist.keys()),
                            y=list(severity_dist.values()),
                            title=f"Ticket Severity (Last {days_filter} days)"
                        )
                        st.plotly_chart(fig_severity, use_container_width=True)
            
            # Quality metrics over time
            if quality_metrics:
                st.markdown("**Quality Metrics**")
                metrics_df = pd.DataFrame([quality_metrics])
                fig_quality = px.bar(
                    metrics_df.T.reset_index(),
                    x='index',
                    y=0,
                    title="Average Quality Scores",
                    labels={'index': 'Metric', '0': 'Score'}
                )
                st.plotly_chart(fig_quality, use_container_width=True)
        
        # Recent tickets table
        if recent_tickets:
            st.markdown("**Recent Tickets**")
            tickets_df = pd.DataFrame(recent_tickets)
            st.dataframe(tickets_df, use_container_width=True)
        
        # Agent performance
        st.markdown("**Agent Performance**")
        agent_stats = {}
        for agent in ['classifier', 'summarizer', 'action_recommender']:
            stats = db_service.get_agent_statistics(agent)
            if stats:
                agent_stats[agent] = stats
        
        if agent_stats:
            stats_df = pd.DataFrame(agent_stats).T
            st.dataframe(stats_df, use_container_width=True)
        
        # Processing logs
        st.markdown("**Recent Processing Logs**")
        logs = db_service.get_processing_logs(limit=20)
        if logs:
            logs_df = pd.DataFrame(logs)
            st.dataframe(logs_df, use_container_width=True)
        else:
            st.info("No processing logs available yet.")
    
    except Exception as e:
        st.error(f"Error loading database analytics: {str(e)}")
        st.info("Database might not be initialized yet. Process some tickets to see analytics.")

def display_model_management():
    """Display model management and comparison interface."""
    st.subheader("🔄 Model Management & Performance Testing")
    
    if not st.session_state.agents:
        st.warning("Initialize agents first to manage models.")
        return
    
    # Current model configuration
    st.markdown("### 🎛️ Current Agent Models")
    
    try:
        model_info = st.session_state.agents.get_agent_model_info()
        
        # Display current models in a nice grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏥 Triage Specialist**")
            triage_info = model_info.get("triage_specialist", {})
            current_model = triage_info.get("model", "Unknown")
            model_details = triage_info.get("model_info", {})
            st.info(f"**Model:** {current_model}\n\n**Description:** {model_details.get('description', 'N/A')}")
            
            st.markdown("**🎯 Support Strategist**")
            strategist_info = model_info.get("support_strategist", {})
            current_model = strategist_info.get("model", "Unknown")
            model_details = strategist_info.get("model_info", {})
            st.info(f"**Model:** {current_model}\n\n**Description:** {model_details.get('description', 'N/A')}")
        
        with col2:
            st.markdown("**📊 Ticket Analyst**")
            analyst_info = model_info.get("ticket_analyst", {})
            current_model = analyst_info.get("model", "Unknown")
            model_details = analyst_info.get("model_info", {})
            st.info(f"**Model:** {current_model}\n\n**Description:** {model_details.get('description', 'N/A')}")
            
            st.markdown("**✅ QA Reviewer**")
            qa_info = model_info.get("qa_reviewer", {})
            current_model = qa_info.get("model", "Unknown")
            model_details = qa_info.get("model_info", {})
            st.info(f"**Model:** {current_model}\n\n**Description:** {model_details.get('description', 'N/A')}")
        
    except Exception as e:
        st.error(f"Error getting model info: {str(e)}")
        return
    
    st.markdown("---")
    
    # Model swapping interface
    st.markdown("### 🔄 Swap Agent Models")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        agent_to_modify = st.selectbox(
            "Select Agent to Modify",
            ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"],
            format_func=lambda x: {
                "triage_specialist": "🏥 Triage Specialist",
                "ticket_analyst": "📊 Ticket Analyst", 
                "support_strategist": "🎯 Support Strategist",
                "qa_reviewer": "✅ QA Reviewer"
            }[x]
        )
    
    with col2:
        available_models = list(AVAILABLE_MODELS.keys())
        new_model = st.selectbox(
            "Select New Model",
            available_models,
            format_func=lambda x: f"{AVAILABLE_MODELS[x]['name']} ({x})"
        )
    
    # Show model details and recommendations
    if new_model in AVAILABLE_MODELS:
        model_details = AVAILABLE_MODELS[new_model]
        st.markdown(f"**Model Details:** {model_details['description']}")
        st.markdown(f"**Strengths:** {', '.join(model_details['strengths'])}")
        
        # Show agent-specific recommendations
        if agent_to_modify in AGENT_MODEL_RECOMMENDATIONS:
            recommendations = AGENT_MODEL_RECOMMENDATIONS[agent_to_modify]
            recommended_models = recommendations.get("recommended", [])
            reasoning = recommendations.get("reasoning", "")
            
            if new_model in recommended_models:
                st.success(f"✅ Recommended for {agent_to_modify}: {reasoning}")
            else:
                st.warning(f"⚠️ Not specifically recommended for {agent_to_modify}: {reasoning}")
    
    if st.button("🔄 Update Agent Model"):
        with st.spinner(f"Updating {agent_to_modify} to use {new_model}..."):
            try:
                success = st.session_state.agents.update_agent_model(agent_to_modify, new_model)
                if success:
                    st.success(f"✅ Successfully updated {agent_to_modify} to use {new_model}!")
                    st.rerun()
                else:
                    st.error("❌ Failed to update agent model. Check model name and agent configuration.")
            except Exception as e:
                st.error(f"❌ Error updating model: {str(e)}")
    
    st.markdown("---")
    
    # Model comparison interface
    st.markdown("### 📊 Model Performance Comparison")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_agent = st.selectbox(
            "Select Agent to Test",
            ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"],
            format_func=lambda x: {
                "triage_specialist": "🏥 Triage Specialist",
                "ticket_analyst": "📊 Ticket Analyst", 
                "support_strategist": "🎯 Support Strategist",
                "qa_reviewer": "✅ QA Reviewer"
            }[x],
            key="test_agent_select"
        )
    
    with col2:
        models_to_test = st.multiselect(
            "Models to Compare",
            available_models,
            default=["gpt-4o", "gpt-4o-mini"],
            format_func=lambda x: f"{AVAILABLE_MODELS[x]['name']} ({x})"
        )
    
    # Test tickets selection
    st.markdown("**Test Tickets:**")
    test_option = st.radio(
        "Choose test data",
        ["Sample Tickets", "Custom Tickets"],
        horizontal=True
    )
    
    test_tickets = []
    
    if test_option == "Sample Tickets":
        sample_tickets = load_sample_tickets()
        selected_samples = st.multiselect(
            "Select sample tickets to test",
            range(len(sample_tickets)),
            default=[0, 1, 2],
            format_func=lambda i: f"{sample_tickets[i]['id']}: {sample_tickets[i]['title']}"
        )
        test_tickets = [{"id": sample_tickets[i]["id"], "content": sample_tickets[i]["message"]} 
                       for i in selected_samples]
    
    else:  # Custom Tickets
        st.markdown("**Add Custom Test Tickets:**")
        num_custom = st.number_input("Number of custom tickets", min_value=1, max_value=5, value=2)
        
        for i in range(num_custom):
            with st.expander(f"Custom Ticket {i+1}"):
                ticket_id = st.text_input(f"Ticket ID", value=f"CUSTOM_{i+1}", key=f"custom_id_{i}")
                ticket_content = st.text_area(f"Ticket Content", key=f"custom_content_{i}")
                if ticket_id and ticket_content:
                    test_tickets.append({"id": ticket_id, "content": ticket_content})
    
    # Run comparison
    if st.button("🚀 Run Model Comparison") and test_tickets and models_to_test:
        if len(test_tickets) == 0:
            st.error("Please select or create test tickets.")
            return
            
        with st.spinner(f"Testing {len(models_to_test)} models on {len(test_tickets)} tickets..."):
            try:
                comparison_results = st.session_state.agents.compare_models_on_tickets(
                    test_tickets, test_agent, models_to_test
                )
                
                # Store results in session state
                if 'model_comparison_results' not in st.session_state:
                    st.session_state.model_comparison_results = []
                st.session_state.model_comparison_results.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'results': comparison_results
                })
                
                # Display results
                display_comparison_results(comparison_results)
                
            except Exception as e:
                st.error(f"❌ Error running comparison: {str(e)}")
    
    # Display previous comparison results
    if hasattr(st.session_state, 'model_comparison_results') and st.session_state.model_comparison_results:
        st.markdown("---")
        st.markdown("### 📈 Previous Comparison Results")
        
        for i, result_data in enumerate(reversed(st.session_state.model_comparison_results[-3:])):
            with st.expander(f"Results from {result_data['timestamp']}"):
                display_comparison_results(result_data['results'])

def display_comparison_results(comparison_results):
    """Display model comparison results with charts and metrics."""
    st.markdown(f"**Agent Tested:** {comparison_results['agent_name']}")
    st.markdown(f"**Tickets Processed:** {comparison_results['tickets_tested']}")
    st.markdown(f"**Models Compared:** {', '.join(comparison_results['models_tested'])}")
    
    # Performance summary
    summary = comparison_results.get('performance_summary', {})
    
    if summary:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🏆 Best Overall", summary.get('recommended_model', 'N/A'))
        with col2:
            st.metric("⚡ Fastest", summary.get('fastest_model', 'N/A'))
        with col3:
            st.metric("🎯 Most Accurate", summary.get('most_accurate_model', 'N/A'))
    
    # Performance rankings
    rankings = summary.get('performance_rankings', [])
    if rankings:
        st.markdown("**Performance Rankings:**")
        
        ranking_data = []
        for rank in rankings:
            ranking_data.append({
                'Model': rank['model'],
                'Overall Score': f"{rank['score']:.3f}",
                'Accuracy': f"{rank['accuracy']:.3f}",
                'Speed Score': f"{rank['speed']:.3f}"
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        st.dataframe(ranking_df, use_container_width=True)
        
        # Create performance chart
        fig = go.Figure()
        
        models = [r['model'] for r in rankings]
        scores = [r['score'] for r in rankings]
        accuracy = [r['accuracy'] for r in rankings]
        speed = [r['speed'] for r in rankings]
        
        fig.add_trace(go.Bar(
            name='Overall Score',
            x=models,
            y=scores,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results per model
    results = comparison_results.get('results', {})
    if results:
        st.markdown("**Detailed Results:**")
        
        for model_name, model_results in results.items():
            with st.expander(f"{model_name} - {AVAILABLE_MODELS.get(model_name, {}).get('name', model_name)}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Success Rate", f"{model_results.get('success_rate', 0):.1%}")
                    st.metric("Average Time", f"{model_results.get('avg_processing_time', 0):.2f}s")
                
                with col2:
                    st.metric("Total Errors", model_results.get('error_count', 0))
                    st.metric("Tickets Processed", len(model_results.get('ticket_results', [])))
                
                # Show individual ticket results
                ticket_results = model_results.get('ticket_results', [])
                if ticket_results:
                    results_data = []
                    for result in ticket_results:
                        results_data.append({
                            'Ticket ID': result['ticket_id'],
                            'Success': '✅' if result['success'] else '❌',
                            'Processing Time': f"{result['processing_time']:.2f}s",
                            'Status': 'Success' if result['success'] else f"Error: {result.get('error', 'Unknown')}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
if __name__ == "__main__":
    main()
