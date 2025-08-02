# Claude Code Integration Guide

This document provides Claude Code specific guidance, development workflows, and technical context for working with the Support Ticket Summarizer v2.2 project.

## Development Context

This is a sophisticated multi-agent AI system with high-performance parallel processing capabilities, enhanced timing infrastructure, and Python 3.10 compatibility. Focus on maintaining the collaborative intelligence architecture while ensuring proper timing data collection, performance analytics, and concurrent processing efficiency.

## Key Components

### 1. Multi-Agent Architecture (`agents.py`)
- **CollaborativeSupportCrew**: Main orchestration class for agent collaboration with parallel processing
- **Agent Types**: Triage Specialist, Ticket Analyst, Support Strategist, QA Reviewer
- **Parallel Processing**: `process_tickets_parallel()` method with async/await and semaphore controls
- **Data Extraction**: Complex regex patterns for parsing collaborative outputs
- **Quality Assessment**: Integration with DeepEval for metrics evaluation

### 2. Web Interface (`streamlit_app.py`)
- **Interactive Dashboard**: Real-time monitoring of agent processing with parallel controls
- **Parallel Processing UI**: Toggle controls, concurrency sliders, and performance metrics
- **Model Management**: Dynamic switching of AI models per agent
- **Quality Assessment Display**: DeepEval metrics and custom faithfulness scoring
- **Batch Processing**: Enhanced CSV and Kaggle processing with parallel execution

### 3. Database Layer (`database_service.py`, `models.py`)
- **PostgreSQL Integration**: Persistent storage with connection pooling and bulk operations
- **SQLAlchemy Models**: Structured data models for all entities with thread-safe access
- **Analytics Support**: Historical data analysis and performance tracking
- **Bulk Operations**: Optimized batch saves for concurrent processing

### 4. Configuration (`config.py`)
- **Multi-Provider Setup**: OpenAI, Anthropic, Cohere integrations
- **Environment Management**: Secure handling of API keys and database URLs
- **LangSmith Configuration**: Tracing and observability setup

## Recent Critical Fixes (Latest)

### High-Performance Parallel Processing (v2.2 - Current Priority)
- **Async Parallel Architecture**: Complete async/await implementation for concurrent ticket evaluation
- **Configurable Concurrency**: Process 1-10 tickets simultaneously with semaphore-controlled resource management
- **Performance Optimization**: 3-5x speedup with thread pool execution and bulk database operations
- **Resource Management**: Thread-safe database connections with connection pooling
- **User Interface**: Interactive parallel processing controls with real-time performance metrics

### Enhanced Timing System (Previous Release)
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

# Test parallel processing
python -c "
import asyncio
from agents import CollaborativeSupportCrew
crew = CollaborativeSupportCrew()
tickets = [{'id': f'TEST{i}', 'content': f'Test ticket {i}'} for i in range(3)]
results = asyncio.run(crew.process_tickets_parallel(tickets, max_concurrent=2))
print(f'Processed {len(results)} tickets in parallel')
"
```

## Parallel Processing Guide

### Using Parallel Processing
```python
import asyncio
from agents import CollaborativeSupportCrew

# Initialize crew
crew = CollaborativeSupportCrew()

# Prepare tickets for batch processing
tickets = [
    {"id": "TICKET001", "content": "Customer billing issue"},
    {"id": "TICKET002", "content": "Technical support request"},
    {"id": "TICKET003", "content": "Product feature inquiry"}
]

# Process in parallel (up to 5 concurrent)
results = asyncio.run(crew.process_tickets_parallel(tickets, max_concurrent=5))

# Check results
for result in results:
    print(f"Ticket {result['ticket_id']}: {result['processing_status']}")
```

### Performance Optimization Guidelines
1. **Optimal Concurrency**: Use 5-7 concurrent tickets for best balance
2. **API Rate Limits**: Monitor AI provider rate limits and adjust concurrency
3. **Memory Management**: Higher concurrency uses more memory - monitor resource usage
4. **Error Handling**: Failed tickets are isolated - batch continues processing
5. **Database Performance**: Bulk operations provide 70% faster saves

### Parallel Processing Best Practices
- **Small Batches**: Process 10-20 tickets at a time for optimal performance
- **Monitor Resources**: Watch CPU and memory usage during concurrent processing
- **Error Recovery**: Check processing_status for each result to identify failures
- **Database Cleanup**: Use bulk operations for better database performance
- **Progress Tracking**: Use Streamlit UI for real-time progress monitoring

## Debugging Tips

### 1. Parallel Processing Issues
- **AsyncIO Errors**: Check that async functions are properly awaited
- **Concurrency Limits**: Reduce max_concurrent if hitting API rate limits
- **Memory Issues**: Lower concurrency or process in smaller batches
- **Thread Pool Errors**: Ensure proper cleanup and exception handling
- **Database Connection Pool**: Check connection pool exhaustion in logs

### 2. Agent Collaboration Issues
- Check console output for agent interactions and disagreements
- Review Langfuse traces for detailed agent communication
- Verify that consensus building is working properly
- Monitor timing tracker for accurate processing times

### 3. Database Connection Problems
- Ensure PostgreSQL is running and accessible
- Check DATABASE_URL format and credentials
- Verify database schema is properly initialized
- Monitor connection pool usage and thread safety

### 4. Evaluation Metric Problems
- Enable debug logging for DeepEval score extraction
- Check that evaluation results are properly parsed
- Verify custom faithfulness scoring is working
- Monitor bulk evaluation saves for performance

### 5. Model Performance Issues
- Test different AI models for each agent role
- Monitor token usage and response times during parallel processing
- Check model-specific error handling in concurrent scenarios
- Use model comparison tools to benchmark parallel performance

## Best Practices

1. **Always test changes with sample tickets** before deploying
2. **Monitor resource usage** especially with LangSmith connections
3. **Use proper error handling** for all external API calls
4. **Document regex patterns** for future maintenance
5. **Keep evaluation metrics real** - avoid hardcoded fallback values

This guide should help you effectively work with and maintain the Support Ticket Summarizer codebase.