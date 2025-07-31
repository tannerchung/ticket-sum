"""
Utility functions for data loading and logging.
"""

import os
import json
import pandas as pd
import kagglehub
from typing import Dict, Any, List
from tqdm import tqdm

def load_ticket_data() -> pd.DataFrame:
    """
    Load customer support ticket data from Kaggle dataset.
    
    Returns:
        pd.DataFrame: DataFrame containing ticket data
    """
    try:
        print("🔄 Downloading customer support ticket dataset from Kaggle...")
        
        # Download the dataset using kagglehub
        path = kagglehub.dataset_download("suraj520/customer-support-ticket-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # Find CSV files in the downloaded path
        csv_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the downloaded dataset")
        
        # Load the first CSV file found (usually the main dataset)
        data_file = csv_files[0]
        print(f"📊 Loading data from: {data_file}")
        
        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df)} tickets from dataset")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Standardize column names if needed
        column_mapping = {}
        for col in df.columns:
            lower_col = col.lower()
            if 'id' in lower_col and 'ticket' in lower_col:
                column_mapping[col] = 'ticket_id'
            elif 'message' in lower_col or 'content' in lower_col or 'description' in lower_col or 'text' in lower_col:
                column_mapping[col] = 'message'
            elif 'customer' in lower_col and 'id' in lower_col:
                column_mapping[col] = 'customer_id'
            elif 'time' in lower_col or 'date' in lower_col:
                column_mapping[col] = 'timestamp'
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
            print(f"🔄 Renamed columns: {column_mapping}")
        
        # Ensure required columns exist
        if 'ticket_id' not in df.columns:
            df['ticket_id'] = df.index + 1
            print("➕ Added ticket_id column")
        
        if 'message' not in df.columns:
            # Try to find the most likely message column
            text_columns = [col for col in df.columns if df[col].dtype == 'object' and df[col].str.len().mean() > 20]
            if text_columns:
                df['message'] = df[text_columns[0]]
                print(f"➕ Using '{text_columns[0]}' as message column")
            else:
                raise ValueError("No suitable message column found in dataset")
        
        # Remove rows with empty messages
        initial_count = len(df)
        df = df.dropna(subset=['message'])
        df = df[df['message'].str.strip() != '']
        final_count = len(df)
        
        if initial_count != final_count:
            print(f"🧹 Removed {initial_count - final_count} rows with empty messages")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading ticket data: {str(e)}")
        print("📋 Creating sample data for demonstration...")
        
        # Create sample data if real data loading fails
        sample_data = {
            'ticket_id': [1, 2, 3, 4, 5],
            'message': [
                "My account has been charged twice for the same transaction. I need a refund immediately.",
                "The mobile app keeps crashing when I try to upload photos. This is very frustrating.",
                "I love the new features you added! The user interface is much more intuitive now.",
                "How do I change my password? I can't find the option in settings.",
                "Critical security vulnerability found in your API endpoint. Please contact me ASAP."
            ],
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
            'timestamp': ['2024-01-15 10:30:00', '2024-01-15 11:45:00', '2024-01-15 14:20:00', '2024-01-15 16:10:00', '2024-01-15 18:05:00']
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Created sample dataset with {len(df)} tickets")
        return df

def save_results(results: List[Dict[str, Any]], filename: str = "results.json") -> None:
    """
    Save processing results to a JSON file.
    
    Args:
        results: List of processed ticket results
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {str(e)}")

def print_ticket_summary(ticket_data: Dict[str, Any]) -> None:
    """
    Print a formatted summary of processed ticket data.
    
    Args:
        ticket_data: Dictionary containing processed ticket information
    """
    print("\n" + "="*80)
    print(f"🎫 TICKET ID: {ticket_data.get('ticket_id', 'N/A')}")
    print("="*80)
    
    # Original message (truncated)
    message = ticket_data.get('original_message', '')
    if len(message) > 100:
        message = message[:100] + "..."
    print(f"📝 Original Message: {message}")
    
    # Classification results
    classification = ticket_data.get('classification', {})
    print(f"\n🏷️  Classification:")
    print(f"   Intent: {classification.get('intent', 'N/A')}")
    print(f"   Severity: {classification.get('severity', 'N/A')}")
    print(f"   Confidence: {classification.get('confidence', 'N/A')}")
    
    # Summary
    summary = ticket_data.get('summary', '')
    print(f"\n📋 Summary: {summary}")
    
    # Action recommendations
    actions = ticket_data.get('action_recommendation', {})
    print(f"\n🎯 Recommended Actions:")
    print(f"   Primary: {actions.get('primary_action', 'N/A')}")
    print(f"   Priority: {actions.get('priority', 'N/A')}")
    print(f"   Est. Resolution: {actions.get('estimated_resolution_time', 'N/A')}")
    
    if actions.get('secondary_actions'):
        print(f"   Secondary: {', '.join(actions.get('secondary_actions', []))}")
    
    if actions.get('notes'):
        print(f"   Notes: {actions.get('notes', '')}")

def create_progress_bar(total: int, desc: str = "Processing") -> tqdm:
    """
    Create a progress bar for tracking processing.
    
    Args:
        total: Total number of items to process
        desc: Description for the progress bar
        
    Returns:
        tqdm: Progress bar instance
    """
    return tqdm(total=total, desc=desc, unit="ticket", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    
    Returns:
        bool: True if environment is properly configured
    """
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set these variables in your .env file")
        return False
    
    print("✅ Environment validation passed")
    return True
