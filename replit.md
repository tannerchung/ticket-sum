# Support Ticket Summarizer

## Overview
This Python application provides a multi-agent GenAI system built with CrewAI to automate customer support ticket processing. It aims to classify, summarize, and recommend actions for support tickets, leveraging advanced AI models. The project downloads ticket data from Kaggle, processes it through a specialized agent pipeline, and delivers structured outputs with robust logging and tracing. Its vision is to automate and enhance customer support operations, offering significant market potential for efficiency and improved service quality.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Multi-Agent Architecture
The system orchestrates three specialized agents using CrewAI: a ClassifierAgent for intent and severity, a SummarizerAgent for concise content summaries, and an ActionRecommenderAgent for next-step suggestions. Each agent has specific roles, goals, and backstories.

### LLM Integration
The system integrates various large language models, allowing for multi-provider support (OpenAI, Cohere, Anthropic). Agents can be assigned different models, and dynamic model swapping is supported without system restart. A performance testing framework compares models, and agent-specific recommendations are provided.

### Data Processing Pipeline
Customer support ticket datasets are downloaded from Kaggle. The system processes CSV files with automatic column mapping, executes sequential agent processing, and outputs structured results in JSON format.

### Observability and Monitoring
Integration with LangSmith provides comprehensive tracing and logging of LLM calls. Progress bars and terminal output offer real-time monitoring, and processing results are saved to JSON files. Individual agent logging is implemented to track contributions.

### Configuration and Error Handling
Environment-based configuration (`.env` files) and centralized `config.py` manage prompts, API keys, and model settings. The system includes environment validation, robust data loading, and progress tracking.

### Streamlit Web Interface
An interactive web application provides real-time agent status monitoring, sample ticket previews, and LangSmith-style activity logging. It includes DeepEval integration for quality assessment (hallucination, relevancy, faithfulness, accuracy), batch processing capabilities, and visual analytics dashboards using Plotly.

### Database Integration
A PostgreSQL database persistently stores tickets, processing logs, and quality evaluations. It includes data models for support tickets, agent processing logs, quality assessments, and agent status. The database service layer provides analytics, performance metrics, and historical data insights, including real-time agent performance monitoring and historical ticket analysis. Authentic collaboration metrics (disagreement, conflict resolution, consensus building) are also tracked and stored.

### Custom Faithfulness Evaluation
A custom faithfulness evaluation system, powered by GPT-4o, directly compares agent outputs to original ticket content. It assesses classification accuracy, summary factualness, and action recommendation appropriateness, with real-time logging of scores and reasoning.

## External Dependencies

### AI/ML Services
- **OpenAI API**: GPT models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- **Cohere API**: Command models (Command R, Command R+, Command)
- **Anthropic API**: Claude models (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)
- **LangSmith**: Observability and tracing for LLM applications.

### Data Sources
- **Kaggle API**: For downloading customer support ticket datasets.
- **kagglehub**: Python library for programmatic Kaggle dataset access.

### Core Libraries
- **CrewAI**: Multi-agent orchestration framework.
- **LangChain**: LLM application framework.
- **pandas**: Data manipulation and CSV processing.
- **python-dotenv**: Environment variable management.
- **tqdm**: Progress bar functionality.

### Development Tools
- Environment variables for API key management.
- JSON output for structured data storage.
- CSV file processing for ticket data ingestion.