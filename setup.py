"""
Setup script for Support Ticket Summarizer deployment
This bypasses uv and uses pip directly for Replit deployment
"""
from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="support-ticket-summarizer",
    version="2.0.0",
    description="Multi-Agent AI System for Customer Support Automation",
    author="AI Agent",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.11",
    entry_points={
        'console_scripts': [
            'start-app=main:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.11",
    ],
)