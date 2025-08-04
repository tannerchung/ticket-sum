# Support Ticket Summarizer

## Overview
This project provides a multi-agent system designed to summarize support tickets efficiently. It aims to streamline customer service operations by leveraging AI agents for ticket analysis, summarization, and routing. Key capabilities include intelligent session management, quality evaluation of AI outputs, and advanced analytics for agent performance and collaboration. The business vision is to enhance customer support efficiency, reduce response times, and improve the quality of ticket resolutions, thereby boosting customer satisfaction and operational cost-effectiveness.

## User Preferences
*   The user wants the agent to make data-driven decisions based on real performance metrics.
*   The user prefers clear distinctions between technical completion and AI performance quality in experiment evaluations.
*   The user wants to understand agent behavior, including decision-making processes, areas for speed improvement, and impact of prompt changes.
*   The user values insights into collaboration breakdowns and how to optimize multi-agent workflows.

## System Architecture

### UI/UX Decisions
The application features an enhanced UI with:
- Side-by-side visualization of completion rate and quality success rate for experiments.
- Interactive charts showing both completion and quality rates by experiment type.
- An advanced analytics dashboard with production-ready visualizations, including a dedicated "Database Analytics" tab with four sub-categories: Traditional Analytics, Collaboration Intelligence, Cost Optimization, and Production Observability.

### Technical Implementations
- **AI Model Integration**: Utilizes OpenAI models for core AI functionalities.
- **Multi-Agent System**: Built using CrewAI for orchestrating multiple AI agents to process and summarize tickets.
- **Experimentation Platform**: Supports A/B testing and evaluation of different AI models and configurations with clear success metrics.
- **DeepEval Integration**: Provides authentic hallucination detection, relevancy scoring, and custom faithfulness evaluation for AI outputs.
- **Session Management**: Implements intelligent session IDs where individual tickets get unique session IDs, and batch processing shares a session per batch for better Langfuse organization.
- **Security Hardening**: All subprocess calls use comprehensively validated static constant arrays, with runtime validation and timeout protection.

### Feature Specifications
- **Ticket Summarization**: Core functionality to condense support tickets into concise summaries.
- **Quality Metrics**: Implements dual success metrics: "Completion Rate" (technical success) and "Quality Success Rate" (performance success, accuracy > 70%).
- **Advanced Analytics Engine**: Provides sophisticated analytics for collaboration intelligence, cost-quality optimization, and production observability.
    - **Collaboration Intelligence**: Measures authentic disagreement, information flow fidelity, and agent influence.
    - **Cost-Quality Optimization**: Offers intelligent model selection and real-time cost efficiency analysis.
    - **Production Observability**: Includes anomaly detection and recommendations for performance issues.
- **Advanced Collaboration Metrics**: Analyzes agent specialization, communication efficiency, workflow optimization, and emergent behavior patterns.

### System Design Choices
- **Modular Design**: Separates concerns into distinct Python modules (e.g., `advanced_analytics.py`, `analytics_dashboard.py`, `collaboration_metrics.py`).
- **Database-Centric**: Relies on PostgreSQL for storing experiment results, analytics data, and application state.
- **Streamlit Frontend**: Uses Streamlit for the user interface, providing interactive controls and data visualization.
- **Robust Deployment**: Supports multiple entry points (`main.py`, `app.py`, `server.py`, `streamlit_app.py`) for flexible deployment configurations, including health checks.
- **Python Version**: Requires Python 3.10+ for compatibility with CrewAI.
- **Deployment Configuration**: Uses Autoscale deployment target with pip-based builds to avoid uv binary conflicts. Standard requirements.txt for dependency management.

## External Dependencies
- **OpenAI API**: For accessing AI models and capabilities.
- **PostgreSQL**: Primary database for application data, experiment results, and analytics.
- **Langfuse Cloud**: For comprehensive external monitoring, tracing, and observability of AI agent interactions using OpenInference instrumentation.
- **CrewAI**: Framework for building and orchestrating multi-agent systems.
- **Streamlit**: For building the web-based user interface.
- **Kaggle API**: Optional integration for data access or model fine-tuning.
- **DeepEval**: For evaluating the quality and performance of AI agent outputs.
- **Numpy**: For numerical operations, with explicit type conversion for PostgreSQL compatibility.
- **Cohere**: Version 5.12.0 for langchain-cohere compatibility.