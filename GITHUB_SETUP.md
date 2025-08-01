# GitHub Upload Instructions

This guide will help you upload the Support Ticket Summarizer project to GitHub.

## Prerequisites

1. **GitHub Account**: Make sure you have a GitHub account
2. **Git Installed**: Ensure Git is installed on your local machine
3. **GitHub CLI (Optional)**: For easier repository creation

## Method 1: Using GitHub Web Interface (Recommended)

### Step 1: Create New Repository
1. Go to [GitHub.com](https://github.com)
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Fill in the repository details:
   - **Repository name**: `support-ticket-summarizer`
   - **Description**: `Multi-agent AI system for intelligent customer support ticket processing`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

### Step 2: Upload Files
1. After creating the repository, GitHub will show upload instructions
2. Download all project files from Replit to your local machine
3. Follow GitHub's instructions to upload the files

## Method 2: Using Git Command Line

### Step 1: Create Repository on GitHub
Follow Step 1 from Method 1 above.

### Step 2: Clone and Upload
```bash
# Clone your empty repository
git clone https://github.com/YOUR_USERNAME/support-ticket-summarizer.git
cd support-ticket-summarizer

# Copy all project files to this directory
# (Download from Replit and copy here)

# Add all files
git add .

# Commit the files
git commit -m "Initial commit: Multi-agent support ticket summarizer with OpenAI, Anthropic, and Cohere integration"

# Push to GitHub
git push origin main
```

## Method 3: Using GitHub CLI

### Step 1: Install GitHub CLI
```bash
# On macOS
brew install gh

# On Windows (using Chocolatey)
choco install gh

# On Linux
sudo apt install gh
```

### Step 2: Create and Upload
```bash
# Authenticate with GitHub
gh auth login

# Create repository
gh repo create support-ticket-summarizer --public --description "Multi-agent AI system for intelligent customer support ticket processing"

# Clone the repository
git clone https://github.com/YOUR_USERNAME/support-ticket-summarizer.git
cd support-ticket-summarizer

# Copy all project files here
# Add, commit, and push files
git add .
git commit -m "Initial commit: Multi-agent support ticket summarizer"
git push origin main
```

## Files to Include

The following files should be uploaded to GitHub:

### Core Application Files
- `main.py` - Main application entry point
- `streamlit_app.py` - Web interface
- `agents.py` - Multi-agent system implementation
- `config.py` - Configuration and settings
- `database_service.py` - Database operations
- `models.py` - Database models
- `utils.py` - Utility functions
- `demo.py` - Demo and testing functionality

### Configuration Files
- `pyproject.toml` - Python project configuration
- `README.md` - Project documentation
- `.gitignore` - Git ignore rules
- `replit.md` - Project architecture documentation

### Documentation
- `GITHUB_SETUP.md` - This file
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `ALTERNATIVE_DEPLOYMENT.md` - Alternative deployment options

## Files to Exclude (Already in .gitignore)

These files should NOT be uploaded:
- `results_*.json` - Processing results
- `*.lock` - Lock files
- `.env` - Environment variables (contains secrets)
- `attached_assets/` - Temporary assets
- `chromadb-*.lock` - Database locks
- `.replit` - Replit configuration

## Environment Variables Setup

After uploading to GitHub, users will need to set up these environment variables:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key (optional)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=ticket-sum
DATABASE_URL=your_postgresql_database_url
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## Repository Description

Use this description for your GitHub repository:

> Multi-agent AI system for intelligent customer support ticket processing using CrewAI. Features OpenAI, Anthropic, and Cohere integration with dynamic model swapping, real-time analytics, and comprehensive quality assessment. Built with Python, Streamlit, and PostgreSQL.

## Repository Topics/Tags

Add these topics to your repository for better discoverability:
- `ai`
- `machine-learning`
- `multi-agent-system`
- `customer-support`
- `openai`
- `anthropic`
- `crewai`
- `streamlit`
- `postgresql`
- `langchain`
- `python`

## License

Consider adding a license file. Popular choices:
- MIT License (most permissive)
- Apache License 2.0
- GNU General Public License v3.0

## Next Steps After Upload

1. **Add Repository Description**: Add the description and topics mentioned above
2. **Create Issues**: Set up GitHub Issues for bug tracking and feature requests
3. **Set up Actions**: Consider GitHub Actions for CI/CD
4. **Add Collaborators**: If working with a team
5. **Create Releases**: Tag versions of your application

## Troubleshooting

### Large Files
If you encounter issues with large files:
- Check the `.gitignore` is working properly
- Remove any large result files or databases
- Use Git LFS for large files if necessary

### Permission Issues
If you get permission errors:
- Make sure you have write access to the repository
- Check your Git credentials
- Use `git config` to set your username and email

### Merge Conflicts
If there are conflicts:
- This shouldn't happen with a new repository
- If it does, carefully review and resolve conflicts

## Support

For GitHub-specific issues:
- Check [GitHub Docs](https://docs.github.com)
- Visit [GitHub Community](https://github.community)

For project-specific issues:
- Create an issue in your repository
- Check the project documentation in `README.md`