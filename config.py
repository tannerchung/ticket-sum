"""
Configuration module for the support ticket summarizer application.
Contains agent settings, prompt templates, and API setup.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "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")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "ticket-sum")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Kaggle Configuration
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME", "tannerchung")
KAGGLE_KEY = os.getenv("KAGGLE_KEY", "db38b7dbb8cb11cf37e98c2183c97bba")

# Dataset Configuration
KAGGLE_DATASET = "suraj520/customer-support-ticket-dataset"

# Model Configuration
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
LLM_MODEL = "gpt-4o"

# Agent Prompts and Templates
CLASSIFIER_PROMPT = """You are a customer support ticket classifier. Analyze and classify tickets accurately.

INTENT OPTIONS: billing, bug, feedback, feature_request, general_inquiry, technical_support, account_issue, refund_request, complaint, compliment

SEVERITY LEVELS:
- critical: Service outages, security issues, data loss, payment failures
- high: Major functionality broken, urgent business impact, angry customers  
- medium: Minor bugs, general complaints, feature requests with business impact
- low: General questions, minor issues, feedback, compliments

Return ONLY valid JSON (no extra text):
{{"intent": "category_name", "severity": "severity_level", "confidence": 0.95, "reasoning": "Brief explanation"}}

Ticket: {ticket_content}"""

SUMMARIZER_PROMPT = """
You are a customer support ticket summarizer. Create a concise, professional summary of the ticket content.

Your summary should include:
1. Main issue or request
2. Key details and context
3. Customer sentiment (if evident)
4. Any technical details mentioned

Keep the summary under 150 words and focus on actionable information.

Classification Context:
Intent: {intent}
Severity: {severity}

Ticket Content: {ticket_content}

Provide a clear, professional summary:
"""

ACTION_RECOMMENDER_PROMPT = """Customer support action recommender. Recommend next steps based on classification and summary.

ACTIONS: escalate_to_tier2, escalate_to_manager, respond_with_template, request_more_info, immediate_response, route_to_billing, route_to_technical, close_with_solution, schedule_callback

Return ONLY valid JSON (no extra text):
{{"primary_action": "action_name", "secondary_actions": ["action1"], "priority": "high", "estimated_resolution_time": "2-4 hours", "notes": "Context for agent"}}

Intent: {intent}, Severity: {severity}
Summary: {summary}"""

# LangSmith Configuration
def setup_langsmith():
    """Configure LangSmith tracing"""
    if LANGSMITH_TRACING:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        print(f"✅ LangSmith tracing enabled for project: {LANGSMITH_PROJECT}")
    else:
        print("❌ LangSmith tracing disabled")

# Kaggle Configuration
def setup_kaggle():
    """Configure Kaggle API credentials"""
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    print(f"✅ Kaggle configured for user: {KAGGLE_USERNAME}")
