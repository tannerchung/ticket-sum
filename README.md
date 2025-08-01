# Support Ticket Summarizer v2.1

A sophisticated Python-powered multi-agent AI system for intelligent customer support ticket processing, leveraging advanced collaborative intelligence and comprehensive telemetry.

## Overview

This application implements a **collaborative multi-agent GenAI system** for customer support automation using CrewAI. The system features four specialized AI agents (Triage Specialist, Ticket Analyst, Support Strategist, QA Reviewer) that work collaboratively with **authentic consensus building**, **dynamic quality evaluation**, and **comprehensive monitoring** through Langfuse Cloud tracing integration.

## Key Features ✨

### 🤖 Advanced Multi-Agent Architecture
- **Triage Specialist**: Fast, accurate ticket classification with intent and severity analysis
- **Ticket Analyst**: Deep content analysis and comprehensive summarization
- **Support Strategist**: Strategic response planning and escalation decisions
- **QA Reviewer**: Quality validation and consensus building across all agents

### 🔀 Multi-Provider AI Support
- **OpenAI Models**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo for reliable performance
- **Anthropic Models**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus for thoughtful reasoning
- **Cohere Models**: Command R, Command R+, Command for business-focused analysis
- **Dynamic Model Assignment**: Each agent can use different models optimized for their specific tasks

### 🎯 Authentic Collaboration Features
- **Real Consensus Building**: Genuine agent disagreement detection and resolution
- **Custom Faithfulness Scoring**: GPT-4o-based evaluation comparing outputs to source content
- **Dynamic Conflict Resolution**: Tracks actual conflicts between agents and resolution methods
- **Collaboration Metrics**: Authentic measurement of agent agreement and consensus strength

### 🌐 Interactive Web Interface
- **Real-time Dashboard**: Live agent status monitoring with processing indicators
- **Model Management Hub**: Dynamic model swapping per agent without system restart
- **Performance Analytics**: Comparative analysis of different AI models across test tickets
- **Database Analytics**: Historical insights with interactive Plotly charts and trend analysis

### 📊 Comprehensive Monitoring & Observability
- **Langfuse Cloud Integration**: Complete tracing of all agent interactions with OpenInference instrumentation and intelligent session management
- **Dynamic Quality Assessment**: Real-time DeepEval metrics with authentic faithfulness evaluation using GPT-4o
- **Session Management**: Option B implementation - individual sessions for single tickets, shared sessions for batch processing
- **PostgreSQL Analytics**: Persistent storage with performance metrics, collaboration tracking, and historical analysis
- **Real-time Collaboration Tracking**: Live monitoring of agent consensus building and conflict resolution with enhanced telemetry

## Technology Stack 🛠️

- **Multi-Agent Framework**: CrewAI with collaborative task orchestration
- **AI Providers**: OpenAI (GPT-4o), Anthropic (Claude 3.5), Cohere (Command R) with fallback support
- **Database**: PostgreSQL with SQLAlchemy for persistent analytics and collaboration metrics
- **Web Interface**: Streamlit with real-time updates and interactive dashboards
- **Observability**: Langfuse Cloud with OpenInference instrumentation and intelligent session management
- **Quality Assessment**: DeepEval integration with dynamic metrics and custom faithfulness evaluation
- **Data Processing**: pandas, kagglehub for dataset management
- **Visualization**: Plotly for interactive charts and performance analytics

## Version History & Release Journey 📈

### v2.1 - Advanced Telemetry & Quality Assessment (January 31, 2025)
**Critical Migration: LangSmith → Langfuse Cloud + Enhanced Session Management**

#### Major Breaking Changes & Migrations
- 🔄 **LangSmith to Langfuse Migration**: Complete telemetry overhaul due to CrewAI 0.80+ incompatibility with LangSmith callbacks
- 🔧 **OpenInference Instrumentation**: Replaced custom LangSmith integration with OpenTelemetry-based tracing for better CrewAI compatibility
- 📊 **DeepEval Metrics Fix**: Resolved hardcoded evaluation scores, now showing real-time dynamic metrics (Hallucination, Relevancy, Faithfulness)
- 🎯 **Intelligent Session Management**: Implemented batch sessions - individual tickets get unique sessions, batch processing shares session IDs

#### Critical Technical Fixes
- **CrewAI Compatibility Crisis**: LangSmith callback system incompatible with CrewAI 0.80+, causing trace failures
- **DeepEval Integration Issues**: Fixed subscription to get actual dynamic scores instead of placeholder values
- **Session Tracking Logic**: Optimized for Langfuse dashboard organization with batch vs individual processing separation
- **Database Schema Reset**: Complete cleanup for fresh analytics tracking with enhanced collaboration metrics

#### Telemetry Architecture Revolution
- **Langfuse Cloud Integration**: Full OTLP exporter setup with automatic trace capture via OpenInference
- **Per-Run Session IDs**: Individual tickets = unique sessions, batch processing = shared session per batch
- **Enhanced Trace Context**: Comprehensive metadata including processing type, agent count, and collaboration metrics
- **Automatic Instrumentation**: No manual callback management - OpenInference handles all LLM trace capture

#### Quality Assessment Improvements  
- **Dynamic DeepEval Scores**: Real-time hallucination (1.000), relevancy (1.000), faithfulness (0.600) calculation
- **Custom Faithfulness Pipeline**: GPT-4o-based fact-checking with fallback mechanisms for offline scenarios
- **Enhanced Collaboration Tracking**: Authentic disagreement detection and consensus building metrics
- **Database Analytics Refresh**: Clean slate for accurate performance tracking and quality trends

### v2.0 - Collaborative Intelligence Era (Previous)
**Major Architecture Evolution: From Sequential to Collaborative Processing**

#### Core Breakthroughs
- ✅ **Authentic Multi-Agent Collaboration**: Transformed from sequential agent processing to genuine collaborative intelligence with real consensus building
- ✅ **Custom Faithfulness Evaluation**: Revolutionary GPT-4o-based fact-checking system that validates agent outputs against source content
- ✅ **Multi-Provider AI Ecosystem**: Complete integration of OpenAI, Anthropic, and Cohere with dynamic model assignment per agent
- ✅ **LangSmith Integration**: Advanced observability with comprehensive tracing (later migrated to Langfuse in v2.1)
- ✅ **Real-Time Collaboration Metrics**: Authentic disagreement detection, conflict resolution tracking, and consensus strength measurement

#### Technical Innovations
- **Dynamic Model Management**: Hot-swappable AI models per agent without system restart
- **Collaborative Memory System**: Shared context and knowledge building across agent interactions
- **Performance Analytics Framework**: Comprehensive model comparison with weighted scoring (70% accuracy, 30% speed)
- **Database Analytics Engine**: Historical insights with interactive Plotly visualizations
- **Robust Error Handling**: JSON serialization fixes and database constraint validation

#### Latest Enhancements (2025)
- **Enhanced Performance Analytics**: Real processing time tracking for accurate agent performance monitoring
- **Improved Accuracy**: Better extraction of severity levels, priorities, and action recommendations
- **Quality Assessment**: Real-time evaluation metrics with authentic DeepEval integration
- **Multi-Model Support**: Seamless switching between OpenAI, Anthropic, and Cohere models

### v1.5 - Enhanced Processing & Monitoring
**Foundation Building: Observability and Quality Assessment**

#### Key Additions
- 🔍 **Langfuse Integration**: Advanced tracing and logging with OpenInference instrumentation for LLM interactions
- 📊 **PostgreSQL Database**: Persistent storage for tickets, processing logs, and evaluations
- 🎯 **Quality Metrics**: Basic DeepEval integration for response quality assessment
- 📈 **Streamlit Interface**: Interactive web dashboard for real-time monitoring
- 🗃️ **Kaggle Integration**: Automated dataset downloading and processing capabilities

#### Technical Improvements
- **Database Schema Design**: Comprehensive models for tickets, logs, and agent status
- **Progress Tracking**: Real-time updates during ticket processing workflows
- **Error Handling**: Initial robust error management and logging systems
- **Configuration Management**: Centralized config system with environment validation

### v1.0 - Multi-Agent Foundation
**Genesis: Sequential Multi-Agent Architecture**

#### Original Implementation
- 🤖 **Four-Agent System**: Initial CrewAI implementation with specialized roles
  - **Classifier Agent**: Intent and severity determination
  - **Summarizer Agent**: Content summarization and key problem identification
  - **Action Recommender Agent**: Next steps and escalation recommendations
  - **QA Agent**: Basic output validation and consistency checking

#### Core Features
- **Sequential Processing**: Linear workflow with agents building on previous outputs
- **OpenAI Integration**: Single-provider implementation using GPT-4o models
- **CSV Processing**: Basic batch processing capabilities for support ticket datasets
- **JSON Output**: Structured results with classification, summary, and recommendations
- **Environment Configuration**: Basic API key management and validation

#### Initial Architecture
- **CrewAI Framework**: Multi-agent orchestration for customer support automation
- **Python-Based**: Core implementation using pandas for data processing
- **Terminal Interface**: Command-line execution with progress indicators
- **File-Based Output**: JSON results saved to local files for analysis

### Development Milestones & Evolution

| Version | Key Innovation | Impact |
|---------|---------------|---------|
| **v1.0** | Multi-agent architecture foundation | Established specialized AI roles for support automation |
| **v1.5** | Observability & persistence layer | Added monitoring, database storage, and quality metrics |
| **v2.0** | Collaborative intelligence breakthrough | Revolutionary authentic agent collaboration with real consensus |

### Technical Evolution Timeline

```
v1.0: Sequential Agents → v1.5: Monitored Processing → v2.0: Collaborative Intelligence
   ↓                           ↓                              ↓
Basic AI Pipeline          Enhanced Observability        Authentic Collaboration
Single Provider           Database Integration           Multi-Provider Ecosystem
File Output              Real-time Monitoring           Dynamic Model Management
Manual Processing        Quality Assessment             Consensus Building
```

This journey represents the evolution from basic multi-agent processing to sophisticated collaborative AI systems capable of genuine consensus building and authentic quality evaluation.

## Quick Start

### Prerequisites
- Python 3.10+ (3.10.18 recommended)
- PostgreSQL database
- OpenAI API key
- LangSmith API key (optional, for tracing)
- Kaggle API credentials (optional, for datasets)

### Environment Variables
Create a `.env` file with:
```bash
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_postgresql_url
LANGSMITH_API_KEY=your_langsmith_key  # Optional
KAGGLE_USERNAME=your_kaggle_username  # Optional
KAGGLE_KEY=your_kaggle_key           # Optional
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ticket-sum.git
cd ticket-sum

# Activate Python 3.10 environment
source venv310/bin/activate

# Install dependencies
pip install -r requirements-py310.txt

# Run the Streamlit interface
streamlit run streamlit_app.py --server.port 5000

# Or run batch processing
python main.py
```

## 📊 Usage

### Web Interface
1. Launch the Streamlit application
2. Select sample tickets or enter custom support requests
3. Watch agents collaborate in real-time through the monitoring dashboard
4. Review detailed analysis, action plans, and quality assessments

### Batch Processing
```python
from agents import CollaborativeSupportCrew
from utils import load_ticket_data

# Initialize collaborative crew
crew = CollaborativeSupportCrew()

# Process single ticket
result = crew.process_ticket("TICKET001", "Customer message here")

# Process batch from CSV or Kaggle dataset
df = load_ticket_data()
results = []
for _, row in df.iterrows():
    result = crew.process_ticket(row['ticket_id'], row['message'])
    results.append(result)
```

## 🏗 Architecture

### Collaborative Workflow
1. **Triage Specialist** performs initial classification and severity assessment
2. **Ticket Analyst** conducts deep analysis and generates comprehensive summaries
3. **Support Strategist** develops action plans and strategic recommendations
4. **QA Reviewer** validates outputs and ensures consensus across all agents

### Data Flow
- Tickets → Multi-agent processing → Database storage → Analytics dashboard
- Real-time status updates and collaboration monitoring
- Comprehensive logging with LangSmith integration
- Quality assessment with DeepEval metrics

### Database Schema
- `support_tickets`: Processed ticket data and classifications
- `processing_logs`: Detailed agent activity logs
- `quality_evaluations`: DeepEval assessment results
- `agent_status`: Real-time agent performance metrics

## 📈 Monitoring & Analytics

- **Agent Performance**: Success rates, processing times, collaboration metrics
- **Ticket Analysis**: Intent distribution, severity patterns, resolution trends
- **Quality Metrics**: Hallucination detection, relevancy scoring, accuracy assessment
- **System Health**: Database analytics, error tracking, performance monitoring

## 🚀 Deployment

The application is designed for flexible deployment with automatic environment configuration:

- **Replit**: Automatic setup with integrated database and secrets management
- **Local Development**: Full PostgreSQL and environment variable configuration
- **Cloud Platforms**: Compatible with major cloud providers (AWS, GCP, Azure)
- **Docker**: Containerized deployment ready (Dockerfile included)

Ensure all environment variables are properly configured and PostgreSQL database is accessible.

## 🤝 Contributing

This project demonstrates advanced multi-agent AI orchestration with practical customer support applications. Contributions are welcome for:

- **Additional AI Provider Integrations**: New model providers and capabilities
- **Enhanced Quality Metrics**: Advanced evaluation frameworks and scoring systems
- **Agent Specializations**: New agent types and collaborative patterns
- **Performance Optimizations**: Speed improvements and resource efficiency
- **Integration Extensions**: CRM systems, ticketing platforms, and workflow tools

## 📝 License

This project is available under standard open source licensing terms for educational and demonstration purposes, showcasing advanced AI agent collaboration techniques in customer support automation.
