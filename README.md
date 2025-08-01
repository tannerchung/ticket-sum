# Support Ticket Summarizer

A sophisticated Python-powered multi-agent AI system for intelligent customer support ticket processing, featuring enhanced automation, resilient service integration, and user-centric design.

## Overview

This application implements a multi-agent GenAI system for customer support automation using CrewAI. The system processes customer support tickets through four specialized AI agents that work in sequence to classify, summarize, and recommend actions for each ticket.

## Key Features

### Multi-Agent Architecture
- **Triage Specialist**: Analyzes tickets to determine intent and severity levels
- **Ticket Analyst**: Generates detailed summaries of ticket content and key problems
- **Support Strategist**: Recommends next steps like escalation or template responses
- **QA Reviewer**: Reviews and validates the work of other agents

### Multi-Provider AI Support
- **OpenAI Models**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic Models**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus
- **Cohere Models**: Command R, Command R+, Command (with compatibility fallback)

### Interactive Web Interface
- **Streamlit Dashboard**: Real-time monitoring and testing interface
- **Model Management**: Dynamic model swapping per agent without restart
- **Performance Analytics**: Compare different AI models across test tickets
- **Database Analytics**: Historical analysis with interactive charts

### Comprehensive Monitoring
- **LangSmith Integration**: Detailed tracing and logging for all LLM interactions
- **DeepEval Quality Assessment**: Hallucination, relevancy, faithfulness, and accuracy scoring
- **PostgreSQL Database**: Persistent storage for tickets, logs, and evaluations
- **Real-time Progress Tracking**: Live updates during ticket processing

## Technology Stack

- **Framework**: CrewAI for multi-agent orchestration
- **AI Providers**: OpenAI, Anthropic, Cohere
- **Database**: PostgreSQL with SQLAlchemy
- **Web Interface**: Streamlit
- **Monitoring**: LangSmith, DeepEval
- **Data Processing**: pandas, kagglehub
- **Visualization**: Plotly

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
