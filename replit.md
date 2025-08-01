# Support Ticket Summarizer - Replit Deployment

## Replit-Specific Setup
This document covers deployment and configuration specific to the Replit platform.

## Environment Configuration
Replit automatically handles Python dependencies and provides integrated secrets management.

## Replit Configuration

### Required Secrets
Set up the following secrets in Replit's Secrets tab:
- `OPENAI_API_KEY`: Your OpenAI API key
- `DATABASE_URL`: PostgreSQL connection string (use Replit DB or external)
- `LANGFUSE_PUBLIC_KEY`: Langfuse Cloud public key for tracing
- `LANGFUSE_SECRET_KEY`: Langfuse Cloud secret key for tracing
- `LANGFUSE_HOST`: Langfuse Cloud host (optional, defaults to https://us.cloud.langfuse.com)
- `KAGGLE_USERNAME`: Kaggle username (optional)
- `KAGGLE_KEY`: Kaggle API key (optional)

### Python Version
Ensure Replit is configured for Python 3.10+ for compatibility with CrewAI.

## Replit Deployment Steps

### 1. Fork/Import Repository
Import the repository into Replit or fork from GitHub.

### 2. Configure Python Version
Ensure Replit uses Python 3.10+ in the `.replit` configuration.

### 3. Install Dependencies
Replit will automatically install from `requirements-py310.txt`.

### 4. Set Environment Secrets
Add all required API keys in Replit's Secrets tab.

### 5. Database Setup
- Use Replit's PostgreSQL addon, or
- Connect to external PostgreSQL database
- Set `DATABASE_URL` in secrets

### 6. Run Application
```bash
streamlit run streamlit_app.py --server.port 5000
```

## Replit-Specific Considerations

### Performance Optimization
- Replit has resource limits; monitor memory usage during large batch processing
- Consider using smaller AI models for cost efficiency in Replit environment

### Port Configuration
Default Streamlit port (5000) works well with Replit's port forwarding.

### File Storage
Use Replit's persistent storage for output files and cached data.

### Monitoring
- Console output is available in Replit's terminal
- Langfuse Cloud integration provides comprehensive external monitoring with OpenInference instrumentation
- Database analytics available through the web interface

## Recent Changes

### Session Management Enhancement (January 31, 2025)
- **Option B Batch Sessions**: Implemented intelligent session ID management:
  - Individual tickets: Each gets unique session ID for granular tracking
  - Batch processing (Kaggle/CSV): All tickets in batch share one session ID
- **Benefits**: Better organization in Langfuse dashboard with logical grouping by processing type
- **Implementation**: `trace_ticket_processing()` accepts optional `batch_session_id` parameter
- **UI Updates**: Streamlit shows batch session ID when processing batches, individual sessions otherwise

### Database Reset (January 31, 2025)
- **Complete Clean**: All tables dropped and recreated with fresh schema
- **Ready State**: Clean database with 0 rows in all data tables, 4 default agent configurations
- **Session Reset**: Fresh application session and cleared telemetry caches

## Troubleshooting

### Common Replit Issues
1. **Dependency Installation**: If packages fail to install, check Python version compatibility
2. **Port Binding**: Ensure Streamlit runs on correct port for Replit forwarding
3. **Environment Variables**: Verify secrets are properly set in Replit's interface
4. **Memory Limits**: Replit has resource constraints for intensive AI processing

### Version Compatibility
- **Python**: 3.10+ required for CrewAI compatibility
- **Cohere**: Version 5.12.0 for langchain-cohere compatibility
- **Database**: PostgreSQL required (SQLite not recommended for production)