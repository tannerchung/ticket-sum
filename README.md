# Support Ticket Summarizer 🎫

A sophisticated Python-powered multi-agent AI system for intelligent customer support ticket processing, with enhanced automation and resilient service integration.

## 🚀 Features

### Multi-Agent Collaboration
- **Four specialized AI agents** working together through CrewAI
- **Real-time collaboration** with agents questioning each other and reaching consensus
- **Memory sharing** between agents for consistent decision-making
- **Conflict resolution** and agreement scoring

### Agent Roles
- 🏥 **Triage Specialist**: Initial classification & routing
- 📊 **Ticket Analyst**: Deep analysis & summary generation
- 🎯 **Support Strategist**: Action planning & strategy development
- ✅ **QA Reviewer**: Quality review & consensus validation

### Advanced Monitoring
- **Real-time agent status monitoring** with collaborative crew dashboard
- **LangSmith integration** for comprehensive tracing and logging
- **DeepEval assessment** for AI output quality evaluation
- **PostgreSQL database** for persistent storage and analytics

### Interactive Interface
- **Streamlit web application** for testing and monitoring
- **Sample ticket library** for quick testing
- **Batch processing** capabilities for CSV uploads
- **Visual analytics** dashboards with interactive charts

## 🛠 Technology Stack

- **CrewAI**: Multi-agent orchestration framework
- **OpenAI GPT-4o**: Natural language processing and classification
- **PostgreSQL**: Persistent data storage and analytics
- **Streamlit**: Interactive web dashboard
- **LangSmith**: LLM application observability and tracing
- **DeepEval**: AI output quality assessment
- **Kaggle API**: Real customer support dataset access

## 🔧 Setup

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