# Installation Guide - Support Ticket Summarizer v2.0

This guide covers different ways to install and run the **collaborative multi-agent Support Ticket Summarizer** with advanced AI integration, real-time monitoring capabilities, and enterprise-grade code quality standards.

## Quick Start (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/tannerchung/support-ticket-summarizer.git
cd support-ticket-summarizer
```

### 2. Set Up Environment
```bash
# Create virtual environment (Python 3.11+ required)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies using uv (recommended) or pip
uv add crewai langsmith langchain-openai langchain-anthropic langchain-cohere pandas python-dotenv tqdm kagglehub streamlit plotly psycopg2-binary sqlalchemy deepeval

# Alternative: using pip
pip install crewai langsmith langchain-openai langchain-anthropic langchain-cohere pandas python-dotenv tqdm kagglehub streamlit plotly psycopg2-binary sqlalchemy deepeval
```

### 3. Configure Environment Variables
Create a `.env` file in the project root:
```bash
# AI Provider API Keys (OpenAI required, others optional)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional for Claude models
COHERE_API_KEY=your_cohere_api_key        # Optional for Cohere models

# LangSmith Tracing (recommended for monitoring)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=default

# Database (PostgreSQL required)
DATABASE_URL=postgresql://user:password@localhost:5432/ticket_db

# Data Sources (optional for Kaggle datasets)
KAGGLE_USERNAME=your_kaggle_username  
KAGGLE_KEY=your_kaggle_api_key
```

### 4. Set Up Database
```bash
# Install PostgreSQL and create database
createdb ticket_db

# The application will automatically create tables on first run
```

### 5. Run the Application
```bash
# Web Interface (recommended for full experience)
streamlit run streamlit_app.py --server.port 5000 --server.address 0.0.0.0

# Command Line Processing
python main.py

# Demo with Kaggle dataset
python demo_kaggle.py
```

## Alternative Installation Methods

### Using Docker (Coming Soon)
Docker support will be added in a future release.

### Using conda
```bash
conda create -n ticket-summarizer python=3.11
conda activate ticket-summarizer
pip install -r requirements.txt  # Use dependencies from README.md
```

### Using uv (Fast Python Package Manager)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
uv run streamlit run streamlit_app.py
```

## API Key Setup

### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account or sign in
3. Navigate to API Keys section
4. Create a new API key
5. Add to your `.env` file

### Anthropic API Key
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an account or sign in
3. Generate an API key
4. Add to your `.env` file

### Cohere API Key (Optional)
1. Go to [Cohere Dashboard](https://dashboard.cohere.ai/)
2. Sign up or sign in
3. Get your API key
4. Add to your `.env` file

### LangSmith API Key
1. Go to [LangSmith](https://smith.langchain.com/)
2. Create an account
3. Generate an API key
4. Add to your `.env` file

### Kaggle API Key
1. Go to [Kaggle](https://www.kaggle.com/)
2. Go to Account Settings
3. Create new API token
4. Add username and key to your `.env` file

## Database Setup Options

### Local PostgreSQL
```bash
# Install PostgreSQL
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql

# Start PostgreSQL service
sudo service postgresql start  # Linux
brew services start postgresql  # macOS

# Create database
createdb ticket_db
```

### Cloud PostgreSQL
- **Supabase**: Free tier available
- **ElephantSQL**: Free tier available  
- **Amazon RDS**: Paid service
- **Google Cloud SQL**: Paid service

Update your `DATABASE_URL` in `.env` with your cloud database connection string.

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you get import errors, make sure all dependencies are installed
pip install --upgrade pip
pip install -r requirements.txt
```

#### Database Connection Errors
- Check your `DATABASE_URL` format
- Ensure PostgreSQL is running
- Verify database exists and permissions are correct

#### API Key Errors
- Double-check all API keys in your `.env` file
- Ensure API keys have proper permissions
- Check API quota limits

#### Port Already in Use
```bash
# If port 8501 is in use, specify a different port
streamlit run streamlit_app.py --server.port 8502
```

### Performance Issues
- Ensure you have sufficient RAM (4GB+ recommended)
- Check your internet connection for API calls
- Monitor API rate limits

### Compatibility Issues
- Python 3.11+ is required
- If using older Python versions, some dependencies may need different versions

## Development Setup

For development work:

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
pytest

# Format code
black .

# Type checking
mypy .
```

## System Requirements & Quality Standards

### Enterprise-Grade Code Quality
This application maintains **zero LSP diagnostic errors** and follows enterprise-grade development standards:

- **Modern Python Standards**: Uses timezone-aware datetime handling (Python 3.11+)
- **Complete Type Safety**: Proper type annotations with Optional types and SecretStr handling
- **Production-Ready Dependencies**: All AI providers (OpenAI, Cohere, Anthropic) with proper error handling
- **Clean Code Standards**: No unused imports, optimized code structure, professional logging

### Recent Quality Improvements (August 1, 2025)
- **Real DeepEval Integration**: Fixed hardcoded evaluation scores to display actual metrics from DeepEval
- **Enhanced Data Extraction**: Improved regex patterns for better parsing of collaborative agent outputs
- **LangSmith Connection Management**: Added proper client cleanup to prevent resource leaks
- **Collaborative Processing**: Better handling of agent disagreements and consensus building
- **Action Plan Enhancement**: Full descriptive text extraction for more meaningful recommendations

### Performance Requirements
- **RAM**: 4GB minimum, 8GB recommended for optimal multi-agent processing
- **CPU**: Multi-core processor recommended for concurrent AI model operations
- **Network**: Stable internet connection for AI provider API calls
- **Storage**: 1GB free space for datasets and processing logs

### AI Provider Rate Limits
- **OpenAI**: GPT-4o has higher rate limits than GPT-3.5-turbo
- **Anthropic**: Claude models have generous rate limits for enterprise use
- **Cohere**: Command models offer competitive rate limits for business applications

## Getting Help

1. Check the [README.md](README.md) for detailed documentation
2. Review the [GITHUB_SETUP.md](GITHUB_SETUP.md) for repository setup
3. Create an issue on GitHub for bugs or feature requests
4. Check the project documentation in `replit.md` for architecture details

## System Requirements

- **Python**: 3.11 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Network**: Internet connection for API calls
- **Database**: PostgreSQL 12 or higher