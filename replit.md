# Support Ticket Summarizer

## Overview

This is a Python application that implements a multi-agent GenAI system for customer support automation using CrewAI. The system processes customer support tickets through three specialized AI agents that work in sequence to classify, summarize, and recommend actions for each ticket. The application downloads customer support ticket data from Kaggle, processes it through the agent pipeline, and provides structured output with comprehensive logging and tracing capabilities.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Multi-Agent Architecture
The system uses CrewAI to orchestrate three specialized agents that work in sequence:
- **ClassifierAgent**: Analyzes tickets to determine intent (billing, bug, feedback, etc.) and severity levels (low, medium, high, critical)
- **SummarizerAgent**: Generates concise summaries of ticket content and key problems
- **ActionRecommenderAgent**: Recommends next steps like escalation, template responses, or requesting more information

### LLM Integration
- Uses OpenAI's GPT-4o model as the underlying language model for all agents
- Configured with low temperature (0.1) for consistent classification results
- Agents are configured with specific roles, goals, and backstories for specialized behavior

### Data Processing Pipeline
- Downloads customer support ticket datasets from Kaggle using kagglehub
- Processes CSV files with automatic column mapping and standardization
- Implements sequential agent processing with progress tracking
- Outputs structured results in JSON format

### Observability and Monitoring
- Integrates LangSmith for comprehensive tracing and logging
- Tracks input/output for each LLM call with metadata
- Provides progress bars and terminal output for real-time monitoring
- Saves processing results to JSON files for analysis

### Configuration Management
- Environment-based configuration using .env files
- Centralized configuration in config.py for prompts, API keys, and model settings
- Modular architecture with separate files for agents, utilities, and configuration

### Error Handling and Validation
- Environment validation to ensure required API keys are present
- Robust data loading with fallback mechanisms
- Progress tracking and user feedback throughout processing

### Streamlit Web Interface
- Interactive web application for testing and monitoring the multi-agent system
- Real-time agent status monitoring with processing indicators
- Sample ticket previews with expected classifications for testing
- LangSmith-style activity logging with input/output traces and metadata
- DeepEval integration for quality assessment (hallucination, relevancy, faithfulness, accuracy)
- Batch processing capabilities for CSV uploads and Kaggle datasets
- Visual analytics dashboards with Plotly charts for evaluation trends

### Database Integration
- PostgreSQL database for persistent storage of tickets, processing logs, and quality evaluations
- Comprehensive data models for support tickets, agent processing logs, quality assessments, and agent status tracking
- Database service layer providing analytics, performance metrics, and historical data insights
- Real-time agent performance monitoring with success rates and processing times
- Historical ticket analysis with intent and severity distribution tracking
- Database analytics dashboard with interactive charts and data exploration tools

### Model Management and Comparison
- **Multi-Provider Support**: Integration with OpenAI (GPT models), Cohere (Command models), and Anthropic (Claude models) for diverse AI capabilities
- **Individual LLM Assignment**: Each agent can use a different language model from any supported provider
- **Dynamic Model Swapping**: Real-time ability to change models for specific agents without system restart
- **Performance Testing Framework**: Comparative analysis of different models across multiple test tickets
- **Agent-Specific Recommendations**: Model recommendations tailored to each agent's role and requirements, including provider-specific strengths
- **Comprehensive Analytics**: Performance metrics including speed, accuracy, and overall effectiveness scores
- **Interactive Interface**: Streamlit-based model management with visual performance comparisons

## Recent Changes (August 1, 2025)

### Authentic Metrics Implementation (Latest)
- **Real Collaboration Tracking**: Implemented genuine agent disagreement detection and conflict resolution metrics
- **Dynamic Consensus Building**: Added authentic time tracking for consensus building processes
- **Conflict Resolution Documentation**: Real-time logging of specific conflicts and resolution methods used
- **Agreement Strength Calculation**: Dynamic calculation based on actual agent outputs and resolution success
- **Database Schema Enhancement**: Added CollaborationMetrics model for persistent storage of authentic metrics
- **No Simulated Data**: All metrics now calculated from actual agent interactions and outputs

### Multi-Provider Model Integration
- **Added Claude (Anthropic) Support**: Integrated Claude 3.5 Sonnet, Claude 3.5 Haiku, and Claude 3 Opus models
- **Enhanced Provider Fallback System**: Graceful degradation when specific AI providers are unavailable
- **Updated Agent Recommendations**: Added Claude models to agent-specific recommendations based on their strengths
- **Improved Error Handling**: Better compatibility management for different langchain package versions

### Model Swapping Functionality Implementation
- Added comprehensive model management system with per-agent LLM assignment
- Implemented dynamic model swapping capability without system restart
- Created performance comparison framework for testing multiple models
- Enhanced Streamlit interface with dedicated model management tab
- Added agent-specific model recommendations and performance analytics

### Architecture Enhancements
- Extended CollaborativeSupportCrew class with model management methods
- Added performance comparison system with weighted scoring (accuracy 70%, speed 30%)
- Implemented model configuration persistence and agent recreation logic
- Enhanced error handling and validation for model updates

## External Dependencies

### AI/ML Services
- **OpenAI API**: Multiple GPT models (GPT-4o, GPT-4o-mini, GPT-3.5-turbo) for natural language processing and classification
- **Cohere API**: Command models (Command R, Command R+, Command) for business reasoning and advanced analysis tasks
- **Anthropic API**: Claude models (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus) for thoughtful reasoning and analysis
- **LangSmith**: Observability and tracing for LLM applications with API endpoint at api.smith.langchain.com

### Data Sources
- **Kaggle API**: Downloads customer support ticket datasets (specifically "suraj520/customer-support-ticket-dataset")
- **kagglehub**: Python library for programmatic Kaggle dataset access

### Core Libraries
- **CrewAI**: Multi-agent orchestration framework for building AI agent systems
- **LangChain**: LLM application framework with OpenAI integration
- **pandas**: Data manipulation and CSV processing
- **python-dotenv**: Environment variable management
- **tqdm**: Progress bar functionality for user feedback

### Development Tools
- Environment variables for API key management
- JSON output for structured data storage
- CSV file processing for ticket data ingestion