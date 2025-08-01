# Support Ticket Summarizer v2.0

A sophisticated Python-powered multi-agent AI system for intelligent customer support ticket processing, leveraging advanced collaborative intelligence and real-time analytics.

## Overview

This application implements a **collaborative multi-agent GenAI system** for customer support automation using CrewAI. The system features four specialized AI agents (Triage Specialist, Ticket Analyst, Support Strategist, QA Reviewer) that work collaboratively with **authentic consensus building**, **custom faithfulness evaluation**, and **comprehensive monitoring** through LangSmith tracing integration.

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
- **LangSmith Integration**: Complete tracing of all agent interactions, memory operations, and LLM calls
- **Custom Quality Assessment**: Authentic faithfulness evaluation using GPT-4o for fact-checking
- **DeepEval Integration**: Hallucination detection, relevancy scoring, and accuracy assessment
- **PostgreSQL Analytics**: Persistent storage with performance metrics, collaboration tracking, and historical analysis
- **Real-time Collaboration Tracking**: Live monitoring of agent consensus building and conflict resolution

## Technology Stack 🛠️

- **Multi-Agent Framework**: CrewAI with collaborative task orchestration
- **AI Providers**: OpenAI (GPT-4o), Anthropic (Claude 3.5), Cohere (Command R) with fallback support
- **Database**: PostgreSQL with SQLAlchemy for persistent analytics and collaboration metrics
- **Web Interface**: Streamlit with real-time updates and interactive dashboards
- **Observability**: LangSmith for comprehensive tracing, custom faithfulness evaluation
- **Quality Assessment**: DeepEval integration with authentic metrics calculation
- **Data Processing**: pandas, kagglehub for dataset management
- **Visualization**: Plotly for interactive charts and performance analytics

## Latest Version Updates (v2.0) 🚀

### Major Enhancements
- ✅ **Authentic Collaboration Metrics**: Real agent disagreement detection and consensus building
- ✅ **Custom Faithfulness Evaluation**: GPT-4o-based fact-checking against source content
- ✅ **Multi-Provider Model Support**: OpenAI, Anthropic, and Cohere integration with dynamic assignment
- ✅ **Enhanced Observability**: Complete LangSmith tracing with memory operations and agent interactions
- ✅ **Robust Error Handling**: JSON serialization fixes and database constraint validation
- ✅ **Performance Analytics**: Comprehensive model comparison and agent performance tracking

## Quick Start

### Prerequisites
- Python 3.11+
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
# Install dependencies
uv add crewai langsmith langchain-openai pandas python-dotenv tqdm kagglehub streamlit plotly psycopg2-binary sqlalchemy deepeval

# Run the Streamlit interface
streamlit run streamlit_app.py --server.port 5000 --server.address 0.0.0.0

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

## 🤝 Contributing

This is a demonstration project showcasing multi-agent AI collaboration for customer support automation. The system emphasizes real agent interaction, consensus building, and comprehensive monitoring.

## 📝 License

This project is for educational and demonstration purposes, showcasing advanced AI agent collaboration techniques.
- API keys for AI providers (OpenAI, Anthropic, optionally Cohere)

### Environment Setup
1. Clone the repository
2. Install dependencies: `uv sync`
3. Set up environment variables:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   COHERE_API_KEY=your_cohere_key (optional)
   LANGSMITH_API_KEY=your_langsmith_key
   DATABASE_URL=your_postgres_url
   KAGGLE_USERNAME=your_kaggle_username
   KAGGLE_KEY=your_kaggle_key
   ```

### Running the Application

#### Web Interface
```bash
streamlit run streamlit_app.py --server.port 5000 --server.address 0.0.0.0
```

#### Command Line Processing
```bash
python main.py
```

## Features

### Model Management
- **Individual Agent Models**: Assign different AI models to each agent
- **Real-time Swapping**: Change models without restarting the system
- **Performance Comparison**: Test multiple models on the same tickets
- **Smart Recommendations**: Agent-specific model suggestions based on task requirements

### Data Sources
- **Kaggle Integration**: Automatically downloads customer support datasets
- **CSV Upload**: Process your own ticket data
- **Sample Data**: Built-in test tickets for demonstration

### Quality Assurance
- **Multi-layer Validation**: Agents review each other's work
- **Consistency Checking**: Automated quality assessments
- **Performance Metrics**: Track accuracy, speed, and effectiveness
- **Historical Analysis**: Trend monitoring and improvement tracking

## Agent Specializations

### Triage Specialist
- **Purpose**: Fast, accurate ticket classification
- **Recommended Models**: GPT-4o, Claude 3.5 Haiku (speed + accuracy)
- **Output**: Intent classification, severity assessment, confidence scores

### Ticket Analyst
- **Purpose**: Deep content analysis and summarization
- **Recommended Models**: GPT-4o, Claude 3.5 Sonnet (analytical depth)
- **Output**: Comprehensive summaries, key issue identification

### Support Strategist
- **Purpose**: Strategic response planning
- **Recommended Models**: GPT-4o, Claude 3 Opus (complex reasoning)
- **Output**: Action recommendations, escalation decisions

### QA Reviewer
- **Purpose**: Quality validation and consistency checking
- **Recommended Models**: GPT-4o, Claude 3.5 Sonnet (thorough review)
- **Output**: Quality assessments, improvement suggestions

## Database Schema

The system maintains comprehensive records including:
- Support tickets with full metadata
- Agent processing logs and timings
- Quality evaluation scores
- Model performance metrics
- Historical trend data

## Deployment

The application is designed for deployment on Replit with automatic environment configuration. For other platforms, ensure all environment variables are properly set and the PostgreSQL database is accessible.

## Contributing

This project demonstrates advanced multi-agent AI orchestration with practical customer support applications. Contributions are welcome for:
- Additional AI provider integrations
- Enhanced quality metrics
- New agent specializations
- Performance optimizations

## License

This project is available under standard open source licensing terms.
