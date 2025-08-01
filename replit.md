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

### Deployment Configuration Fixes (August 1, 2025)
- **Fixed Critical Deployment Issues**: Resolved deployment failures with comprehensive configuration updates
  - **Created `replit.toml`**: Proper deployment configuration with correct run command `["python", "run.py"]` 
  - **Updated Run Command**: Fixed from generic Python execution (`python3 $file`) to proper Streamlit command via run.py
  - **Enhanced Health Checks**: Implemented robust health check endpoints for deployment verification
    - Added health check query parameter support (`?health=true`)
    - Created dedicated `health.py` endpoint for HTTP health checks
    - Updated health check configuration in `replit.toml` with proper timeout and period settings
  - **Workflow Configuration**: Updated workflow to use `python run.py` with port 5000 binding
  - **Verified Application Response**: Confirmed application responds with HTTP 200 on root endpoint `/`
- **Streamlit API Updates**: Fixed deprecated `st.experimental_get_query_params()` to use modern `st.query_params`
- **Deployment Ready**: Application now properly starts and responds to health checks for successful deployment

### Collaboration Metrics Database Fix (August 1, 2025)
- **Fixed Missing Database Saves**: Resolved issue where collaboration metrics were calculated but not saved to database
  - **Root Cause**: `save_collaboration_metrics()` method existed but was not being called in streamlit_app.py
  - **Solution**: Added missing database save call for collaboration metrics in ticket processing workflow
  - **Impact**: Collaboration metrics (disagreements, conflicts, consensus data) now properly saved to `collaboration_metrics` table
  - **Data Captured**: Disagreement counts, conflict identification, resolution methods, consensus building duration, agreement scores
- **Enhanced Monitoring**: Collaboration analytics now available in database dashboard with authentic multi-agent interaction data

### Individual Agent Logging Implementation (August 1, 2025)
- **Fixed Aggregate Logging Issue**: Resolved problem where all agents were logged as single "collaborative_crew" entity
  - **Previous Problem**: Only one log entry per ticket with aggregate data, individual agent contributions lost
  - **New Implementation**: Each agent (triage_specialist, ticket_analyst, support_strategist, qa_reviewer) now logs individually
  - **Data Captured Per Agent**: 
    - Individual input data (specific task description and role)
    - Individual output data (agent's specific response)
    - Agent-specific metadata (model used, provider, temperature, task type)
    - Individual trace IDs for tracking
    - Agent position and task type (classification, analysis, strategy, review)
  - **Database Impact**: Processing logs now show 4+ individual agent entries plus 1 collaborative summary per ticket
- **Improved Traceability**: Can now track exactly which agent contributed what output and with which model/settings

### Documentation Updates (Previous)
- **README.md Version History**: Added comprehensive version journey documentation showing evolution from v1.0 sequential processing to v2.0 collaborative intelligence with detailed milestones and technical innovations
- **README.md v2.0 Update**: Comprehensive revision reflecting major version changes including collaborative multi-agent architecture, authentic metrics, custom faithfulness evaluation, and multi-provider AI support
- **INSTALL.md Corrections**: Fixed repository name from placeholder to actual GitHub repository (tannerchung/support-ticket-summarizer) and updated installation instructions for v2.0 features
- **Evolution Timeline**: Documented the complete technical evolution from basic AI pipeline to sophisticated collaborative systems with consensus building

### Custom Faithfulness Evaluation Implementation (Latest)
- **Authentic Faithfulness Scoring**: Implemented custom faithfulness evaluation that directly compares agent outputs to original ticket content
- **GPT-4o Based Analysis**: Uses OpenAI's GPT-4o to evaluate how well agents stick to facts in the original message
- **Multi-Component Assessment**: Evaluates classification accuracy, summary factualness, and action recommendation appropriateness
- **Fallback Mechanism**: Keyword-based fallback evaluation for cases where AI evaluation fails
- **Real-Time Logging**: Tracks faithfulness scores for each ticket processing with detailed reasoning
- **No Default Scores**: Replaces static default faithfulness scores with dynamic, authentic calculations

### Complete Git Merge Conflict Resolution
- **All Merge Conflicts Resolved**: Systematically removed all Git merge conflict markers from Python files and configuration files
- **Syntax Error Fixes**: Corrected duplicate code sections, indentation errors, and syntax issues in agents.py, database_service.py, config.py, streamlit_app.py, and models.py
- **Application Restoration**: Both Streamlit interface and Support Ticket Summarizer workflows now running successfully
- **System Validation**: Confirmed all workflows operational with proper database initialization and LangSmith integration

### Authentic Metrics Implementation
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