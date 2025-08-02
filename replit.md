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

### v2.1.2 - Security Enhancement (August 2, 2025)

#### Command Injection Vulnerability Fix - COMPLETED ✅
- **Security Issue Fixed**: Potential command injection vulnerability in main.py subprocess.run() fallback mechanism
- **Root Cause**: Improper command array concatenation where `cmd_fallback = [sys.executable, '-m'] + cmd` created malformed command structure
- **Vulnerability**: Dynamic command construction could potentially be exploited if any input were externally controlled
- **Solution**: Replaced dynamic command concatenation with explicit static command array construction
- **Implementation**: 
  - Removed vulnerable line: `cmd_fallback = [sys.executable, '-m'] + cmd`
  - Added secure static command array with explicit argument values
  - All subprocess.run() calls now use properly constructed command arrays with static strings
- **Security Enhancement**: Application now follows secure coding practices for subprocess execution
- **Verification**: Audited all subprocess usage across codebase - run.py already secure with static command construction

### v2.1.1 - Langfuse Session Tracking Enhancement (August 2, 2025)

#### Session ID Implementation Fix - COMPLETED ✅
- **Problem Solved**: Session IDs were not appearing in Langfuse traces despite tracing being active
- **Root Cause**: Incorrect Langfuse trace API usage and conflicting UI elements
- **Solution**: Streamlined approach using OpenInference instrumentation for automatic session tracking
- **Implementation**: 
  - Fixed trace creation to rely on OpenInference instrumentation rather than direct API calls
  - Removed problematic UI button from form context causing Streamlit errors
  - Session IDs now display correctly with full UUID and copy functionality
  - Verified working with successful ticket processing showing session tracking

#### Experiment JSON Serialization Fix - COMPLETED ✅
- **Problem Fixed**: Model comparison experiments failing with "ExperimentType is not JSON serializable" error
- **Root Cause**: ExperimentType enum being passed directly to database without string conversion
- **Solution**: Convert enum to string value before JSON serialization in experiment_manager.py
- **Result**: Model comparison experiments now run successfully without database errors

#### Langfuse Trace API Fix - COMPLETED ✅
- **Problem Fixed**: "Langfuse object has no attribute 'trace'" warnings during processing
- **Root Cause**: Attempting to use non-existent client.trace() method for explicit trace creation
- **Solution**: Removed explicit trace creation, relying entirely on OpenInference automatic instrumentation
- **Result**: Clean trace processing without AttributeError warnings

#### Numpy PostgreSQL Serialization Fix - COMPLETED ✅
- **Problem Fixed**: Experiment database updates failing with "schema 'np' does not exist" error
- **Root Cause**: Numpy float64 values being passed directly to PostgreSQL without type conversion
- **Solution**: Added explicit float() conversion for all numpy values in experiment results
- **Result**: Experiment runs now complete successfully with proper database storage

#### Trace Naming Improvements
- **Issue Fixed**: Generic "completion" and "ToolUsage._use" trace names from OpenInference instrumentation
- **Enhancement**: Better span naming and resource attributes for clearer Langfuse visibility
- **Results**: 
  - Traces now appear as "support-ticket-individual-{ticket_id}" or "support-ticket-batch-{ticket_id}"
  - Enhanced resource attributes include service version, environment, and session context
  - Improved trace organization with detailed metadata

#### Database Migration Completion - VERIFIED ✅
- **Completed**: Full LangSmith → Langfuse column migration
- **Updated Schema**: Replaced `langsmith_run_id` with `langfuse_trace_id`, `langfuse_session_id`, `langfuse_observation_id`
- **Code Updates**: All database service methods and calls updated for new Langfuse parameters
- **Status**: Application successfully processing tickets with proper session tracking and quality evaluation
- **Test Results**: DeepEval showing 100% pass rates, session IDs displaying correctly in UI

### v2.1.0 - Critical Migration & Quality Enhancements (January 31, 2025)

#### LangSmith → Langfuse Migration (BREAKING CHANGE)
- **Migration Reason**: CrewAI 0.80+ incompatibility caused trace capture failures with LangSmith callbacks
- **Solution**: Complete migration to Langfuse Cloud with OpenInference instrumentation
- **Benefits**: Automatic trace capture, no callback conflicts, better CrewAI compatibility
- **Implementation**: OTLP exporter with comprehensive metadata tracking

#### Session Management Enhancement - Option B
- **Intelligent Sessions**: Individual tickets get unique session IDs, batch processing shares session per batch
- **Langfuse Organization**: Better dashboard grouping with logical separation of processing types
- **UI Enhancement**: Streamlit displays batch session IDs during batch processing operations
- **Implementation**: `trace_ticket_processing()` accepts optional `batch_session_id` parameter

#### DeepEval Integration Fixes
- **Problem Fixed**: Hardcoded placeholder values (1.000, 1.000, 0.600) replaced with dynamic calculation
- **Real Metrics**: Authentic hallucination detection, relevancy scoring, and custom faithfulness evaluation
- **Quality Pipeline**: Enhanced GPT-4o-based fact-checking with fallback mechanisms
- **Result**: Genuine quality assessment reflecting actual AI agent performance

#### Database & Schema Reset
- **Complete Clean**: All tables dropped and recreated with enhanced collaboration metrics schema
- **Fresh Analytics**: Clean slate for accurate performance tracking and quality trend analysis
- **Ready State**: 0 rows in data tables, 4 default agent configurations maintained

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