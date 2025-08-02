#!/bin/bash
# Python 3.10 Environment Activation Script
# Use this script to activate the Python 3.10 environment for the Support Ticket Summarizer

echo "🚀 Activating Python 3.10 Environment for Support Ticket Summarizer"
echo "=================================================================="

# Activate the Python 3.10 virtual environment
source venv310/bin/activate

# Verify Python version
echo "🐍 Python Version: $(python --version)"

# Verify key packages
echo "📦 Key Packages:"
echo "  - CrewAI: $(python -c 'import crewai; print(crewai.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  - LangSmith: $(python -c 'import langsmith; print(langsmith.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  - LangChain: $(python -c 'import langchain; print(langchain.__version__)' 2>/dev/null || echo 'Not installed')"

# Show LangSmith project configuration
echo "🔗 LangSmith Configuration:"
echo "  - Project: ticket-sum"
echo "  - Tracing: Enabled"

echo ""
echo "✅ Environment activated successfully!"
echo ""
echo "🚀 Available commands:"
echo "  streamlit run streamlit_app.py --server.port 5000  # Web interface"
echo "  python main.py                                     # Command line processing"
echo "  python demo.py                                     # Demo with sample tickets"
echo "  python demo_kaggle.py                              # Demo with Kaggle dataset"
echo ""
echo "💡 To deactivate, run: deactivate"
echo "==================================================================" 