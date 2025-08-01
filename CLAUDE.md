# Claude Code Integration Guide

This document provides Claude Code specific guidance, development workflows, and technical context for working with the Support Ticket Summarizer v2.0 project.

## Development Context

This is a sophisticated multi-agent AI system with enhanced timing infrastructure and Python 3.10 compatibility. Focus on maintaining the collaborative intelligence architecture while ensuring proper timing data collection and performance analytics.

## Key Components

### 1. Multi-Agent Architecture (`agents.py`)
- **CollaborativeSupportCrew**: Main orchestration class for agent collaboration
- **Agent Types**: Triage Specialist, Ticket Analyst, Support Strategist, QA Reviewer
- **Data Extraction**: Complex regex patterns for parsing collaborative outputs
- **Quality Assessment**: Integration with DeepEval for metrics evaluation

### 2. Web Interface (`streamlit_app.py`)
- **Interactive Dashboard**: Real-time monitoring of agent processing
- **Model Management**: Dynamic switching of AI models per agent
- **Quality Assessment Display**: DeepEval metrics and custom faithfulness scoring
- **LangSmith Integration**: Tracing and logging of agent activities

### 3. Database Layer (`database_service.py`, `models.py`)
- **PostgreSQL Integration**: Persistent storage for tickets, logs, and evaluations
- **SQLAlchemy Models**: Structured data models for all entities
- **Analytics Support**: Historical data analysis and performance tracking

### 4. Configuration (`config.py`)
- **Multi-Provider Setup**: OpenAI, Anthropic, Cohere integrations
- **Environment Management**: Secure handling of API keys and database URLs
- **LangSmith Configuration**: Tracing and observability setup

## Recent Critical Fixes (Latest)

### Enhanced Timing System (Current Priority)
- **AgentTimingTracker**: Thread-safe timing infrastructure replacing hardcoded 0.0 processing times
- **Multi-Priority Timing**: 4-level fallback system (tracker → callbacks → summed → timestamps)
- **Real Database Integration**: Processing times > 0.0 now flow to database correctly
- **Performance Analytics**: Accurate agent timing for monitoring dashboards

### Python 3.10 Compatibility 
- **CrewAI Union Syntax**: Fixed Python 3.9 incompatibility with CrewAI 0.152.0
- **Cohere Integration**: Downgraded to compatible version (5.12.0) for langchain-cohere
- **Virtual Environment**: `venv310/` with all working dependencies locked

### Quality Assessment & Data Extraction
- **Real DeepEval Scores**: Fixed hardcoded evaluation metrics to display actual results
- **Enhanced Regex Patterns**: Better parsing of collaborative agent outputs
- **LangSmith Cleanup**: Proper connection management preventing memory leaks

## Testing and Quality Assurance

### Running Tests
```bash
# Run quality evaluations
python -m pytest tests/ -v

# Run specific test for data extraction
python -m pytest tests/test_extraction.py -v

# Test collaborative processing
python -m pytest tests/test_collaboration.py -v
```

### Code Quality Commands
```bash
# Type checking
mypy agents.py streamlit_app.py

# Linting
ruff check .

# Formatting
ruff format .
```

### Manual Testing
```bash
# Test single ticket processing
python main.py --test-single

# Test collaborative workflow
python main.py --test-collaboration

# Run DeepEval assessment
python main.py --test-evaluation
```

## Development Guidelines

### When Making Changes
1. **Test Locally First**: Always test changes with sample tickets before committing
2. **Check Data Extraction**: Verify that regex patterns work with actual collaborative outputs
3. **Monitor LangSmith**: Ensure tracing connections are properly managed
4. **Validate DeepEval**: Confirm evaluation scores are displaying correctly

### Common Issues and Solutions

#### 1. Hardcoded Evaluation Scores
**Problem**: DeepEval showing static values instead of real metrics
**Solution**: Check `evaluate_with_deepeval()` function in `streamlit_app.py` for proper score extraction

#### 2. Data Extraction Failures
**Problem**: Severity, priority, or actions showing as "classification", "level", or generic codes
**Solution**: Review regex patterns in `_extract_classification_values()` and `_extract_action_plan()` methods

#### 3. LangSmith Connection Issues
**Problem**: Connections not closing properly, potential memory leaks
**Solution**: Ensure all LangSmith client usage includes proper cleanup in finally blocks

#### 4. Collaborative Processing Errors
**Problem**: Agents not reaching consensus or conflicts not being tracked
**Solution**: Check `_track_collaboration_metrics()` and ensure disagreement detection is working

### File Structure
```
ticket-sum/
├── agents.py              # Multi-agent orchestration
├── streamlit_app.py       # Web interface and dashboards
├── database_service.py    # Database operations
├── models.py             # SQLAlchemy data models
├── config.py             # Configuration management
├── utils.py              # Utility functions
├── main.py               # Command-line interface
├── tests/                # Test suite
└── requirements.txt      # Dependencies
```

## Environment Setup

### Required Environment Variables
```bash
# AI Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=... (optional)

# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Observability
LANGSMITH_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ticket-sum

# Data Sources
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

### Python 3.10 Environment Setup
```bash
# Use Python 3.10 environment (REQUIRED)
source venv310/bin/activate

# Install from locked requirements
pip install -r requirements-py310.txt

# Run with Python 3.10
streamlit run streamlit_app.py --server.port 5000

# Test timing integration
python test_timing_simple.py
```

## Debugging Tips

### 1. Agent Collaboration Issues
- Check console output for agent interactions and disagreements
- Review LangSmith traces for detailed agent communication
- Verify that consensus building is working properly

### 2. Database Connection Problems
- Ensure PostgreSQL is running and accessible
- Check DATABASE_URL format and credentials
- Verify database schema is properly initialized

### 3. Evaluation Metric Problems
- Enable debug logging for DeepEval score extraction
- Check that evaluation results are properly parsed
- Verify custom faithfulness scoring is working

### 4. Model Performance Issues
- Test different AI models for each agent role
- Monitor token usage and response times
- Check model-specific error handling

## Best Practices

1. **Always test changes with sample tickets** before deploying
2. **Monitor resource usage** especially with LangSmith connections
3. **Use proper error handling** for all external API calls
4. **Document regex patterns** for future maintenance
5. **Keep evaluation metrics real** - avoid hardcoded fallback values

This guide should help you effectively work with and maintain the Support Ticket Summarizer codebase.