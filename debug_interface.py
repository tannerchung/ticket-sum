"""
Streamlit debug interface for live logging and process monitoring.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from live_logger import live_logger, LogLevel, ProcessStatus

def display_debug_interface():
    """Display the live debug log interface."""
    st.subheader("🔍 Live Debug Console")
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Refresh Logs", key="refresh_logs"):
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Completed", key="clear_completed"):
            live_logger.clear_completed_processes()
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox("Auto Refresh", key="auto_refresh_logs")
    
    with col4:
        st.metric("Active Processes", len(live_logger.get_active_processes()))
    
    # Process control section
    st.markdown("### 🎮 Process Control")
    
    active_processes = live_logger.get_active_processes()
    if active_processes:
        for process in active_processes:
            process_id = process['process_id']
            process_name = process['name']
            status = process['status']
            progress = process['progress']
            
            with st.expander(f"⚡ {process_name} ({process_id})", expanded=True):
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    # Progress bar
                    st.progress(progress, text=f"{progress:.1%} - {process.get('current_step', 'Running...')}")
                
                with col2:
                    status_color = {
                        'RUNNING': '🟢',
                        'PAUSED': '🟡', 
                        'STOPPED': '🔴',
                        'COMPLETED': '✅',
                        'ERROR': '❌'
                    }.get(status, '⚪')
                    st.markdown(f"{status_color} **{status}**")
                
                with col3:
                    if st.button("⏸️ Pause", key=f"pause_{process_id}", disabled=(status != 'RUNNING')):
                        live_logger.pause_process(process_id)
                        st.rerun()
                
                with col4:
                    if st.button("▶️ Resume", key=f"resume_{process_id}", disabled=(status != 'PAUSED')):
                        live_logger.resume_process(process_id)
                        st.rerun()
                
                with col5:
                    if st.button("⏹️ Stop", key=f"stop_{process_id}", disabled=(status in ['STOPPED', 'COMPLETED'])):
                        live_logger.stop_process(process_id)
                        st.rerun()
                
                # Process details
                st.caption(f"Started: {process['start_time']} | Steps: {process.get('completed_steps', 0)}/{process.get('total_steps', 0)}")
    else:
        st.info("No active processes running")
    
    # Log filtering options
    st.markdown("### 📋 Log Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level_filter = st.selectbox(
            "Log Level",
            options=['All'] + [level.value for level in LogLevel],
            key="log_level_filter"
        )
    
    with col2:
        all_processes = ['All'] + [p['process_id'] for p in live_logger.get_processes()]
        process_filter = st.selectbox(
            "Process",
            options=all_processes,
            key="process_filter"
        )
    
    with col3:
        log_limit = st.number_input(
            "Max Logs",
            min_value=10,
            max_value=1000,
            value=100,
            key="log_limit"
        )
    
    # Get filtered logs
    level_filter = None if log_level_filter == 'All' else LogLevel(log_level_filter)
    process_id_filter = None if process_filter == 'All' else process_filter
    
    logs = live_logger.get_logs(
        process_id=process_id_filter,
        level_filter=level_filter,
        limit=int(log_limit)
    )
    
    # Display logs
    st.markdown(f"### 📝 Live Logs ({len(logs)} entries)")
    
    if logs:
        # Create log display with colors
        log_container = st.container()
        
        with log_container:
            # Reverse to show newest first
            for log in reversed(logs[-50:]):  # Show last 50 logs
                level = log['level']
                timestamp = log['timestamp']
                message = log['message']
                process_id = log['process_id']
                
                # Color coding for log levels
                color_map = {
                    'DEBUG': '#6c757d',    # Gray
                    'INFO': '#0d6efd',     # Blue
                    'WARNING': '#fd7e14',  # Orange
                    'ERROR': '#dc3545',    # Red
                    'SUCCESS': '#198754'   # Green
                }
                
                color = color_map.get(level, '#6c757d')
                
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime("%H:%M:%S")
                except:
                    time_str = timestamp[-8:] if len(timestamp) > 8 else timestamp
                
                st.markdown(f"""
                <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 5px; font-family: monospace; font-size: 12px;">
                    <span style="color: #666;">{time_str}</span> 
                    <span style="color: {color}; font-weight: bold;">[{level}]</span> 
                    <span style="color: #888; font-size: 10px;">({process_id})</span>
                    <br>
                    <span style="color: #333;">{message}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Log statistics
        if len(logs) > 0:
            st.markdown("---")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            level_counts = {}
            for log in logs:
                level = log['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            with col1:
                st.metric("DEBUG", level_counts.get('DEBUG', 0))
            with col2:
                st.metric("INFO", level_counts.get('INFO', 0))
            with col3:
                st.metric("WARNING", level_counts.get('WARNING', 0))
            with col4:
                st.metric("ERROR", level_counts.get('ERROR', 0))
            with col5:
                st.metric("SUCCESS", level_counts.get('SUCCESS', 0))
    else:
        st.info("No logs available with current filters")
    
    # Export logs
    if logs and st.button("📥 Export Logs", key="export_logs"):
        df_logs = pd.DataFrame(logs)
        csv = df_logs.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"debug_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Auto refresh
    if auto_refresh:
        import time
        time.sleep(2)
        st.rerun()

def initialize_debug_logging():
    """Initialize debug logging for the application."""
    if 'debug_initialized' not in st.session_state:
        from live_logger import log_info
        log_info("🚀 Support Ticket Summarizer initialized")
        st.session_state.debug_initialized = True

def log_ticket_processing_start(ticket_id: str, ticket_type: str = "individual") -> str:
    """Start logging for ticket processing."""
    process_name = f"Ticket Processing - {ticket_id}"
    process_id = live_logger.start_process(process_name, total_steps=4)
    live_logger.log_info(f"Starting {ticket_type} ticket processing for {ticket_id}", process_id)
    return process_id

def log_ticket_processing_step(process_id: str, step: str, step_number: int = None):
    """Log a ticket processing step."""
    if step_number:
        live_logger.update_process(process_id, current_step=step, completed_steps=step_number)
    live_logger.log_info(f"Step: {step}", process_id)

def log_ticket_processing_complete(process_id: str, success: bool = True, results: Dict[str, Any] = None):
    """Complete ticket processing logging."""
    live_logger.complete_process(process_id, success)
    if results:
        live_logger.log_success(f"Processing completed. Results: {len(results)} items processed", process_id)
    else:
        live_logger.log_success("Processing completed successfully", process_id)

def log_experiment_start(experiment_name: str, experiment_type: str) -> str:
    """Start logging for an experiment."""
    process_name = f"Experiment - {experiment_name}"
    process_id = live_logger.start_process(process_name, total_steps=5)
    live_logger.log_info(f"Starting {experiment_type} experiment: {experiment_name}", process_id)
    return process_id

def log_experiment_step(process_id: str, step: str, step_number: int = None):
    """Log an experiment step."""
    if step_number:
        live_logger.update_process(process_id, current_step=step, completed_steps=step_number)
    live_logger.log_info(f"Experiment step: {step}", process_id)

def log_experiment_complete(process_id: str, success: bool = True, results: Dict[str, Any] = None):
    """Complete experiment logging."""
    live_logger.complete_process(process_id, success)
    if results:
        live_logger.log_success(f"Experiment completed. Results: {results.get('runs_completed', 0)} runs", process_id)
    else:
        live_logger.log_success("Experiment completed successfully", process_id)