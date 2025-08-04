"""
Database models for the Support Ticket Summarizer system.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class SupportTicket(Base):
    """Model for storing support tickets and their processing results."""
    __tablename__ = 'support_tickets'
    
    id = Column(Integer, primary_key=True)
    ticket_id = Column(String(100), unique=True, nullable=False)
    original_message = Column(Text, nullable=False)
    
    # Classification results
    intent = Column(String(100))
    severity = Column(String(50))
    classification_confidence = Column(Float)
    
    # Summary and recommendations
    summary = Column(Text)
    action_recommendation = Column(JSON)
    
    # Processing metadata
    processing_status = Column(String(50), default='pending')
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)
    
    # Source information
    source = Column(String(100), default='manual')  # 'manual', 'kaggle', 'csv_upload'
    
    def __repr__(self):
        return f"<SupportTicket(ticket_id='{self.ticket_id}', intent='{self.intent}', severity='{self.severity}')>"

class ProcessingLog(Base):
    """Model for logging agent processing activities."""
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True)
    ticket_id = Column(String(100), nullable=False)
    agent_name = Column(String(100), nullable=False)  # 'triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer'
    
    # Processing details
    input_data = Column(JSON)
    output_data = Column(JSON)
    processing_metadata = Column(JSON)
    
    # Timing and status
    processing_time = Column(Float)  # in seconds
    status = Column(String(50))  # 'success', 'error', 'timeout'
    error_message = Column(Text)
    
    # Langfuse Tracing
    trace_id = Column(String(100))
    langfuse_trace_id = Column(String(100))
    langfuse_session_id = Column(String(100))
    langfuse_observation_id = Column(String(100))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ProcessingLog(ticket_id='{self.ticket_id}', agent='{self.agent_name}', status='{self.status}')>"

class QualityEvaluation(Base):
    """Model for storing DeepEval quality assessment results."""
    __tablename__ = 'quality_evaluations'
    
    id = Column(Integer, primary_key=True)
    ticket_id = Column(String(100), nullable=False)
    
    # Quality metrics
    hallucination_score = Column(Float)
    relevancy_score = Column(Float)
    faithfulness_score = Column(Float)
    overall_accuracy = Column(Float)
    
    # Evaluation metadata
    evaluation_model = Column(String(100), default='deepeval')
    evaluation_version = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<QualityEvaluation(ticket_id='{self.ticket_id}', accuracy={self.overall_accuracy})>"

class AgentStatus(Base):
    """Model for tracking real-time agent status."""
    __tablename__ = 'agent_status'
    
    id = Column(Integer, primary_key=True)
    agent_name = Column(String(100), unique=True, nullable=False)
    
    status = Column(String(50), default='inactive')  # 'active', 'inactive', 'processing', 'error'
    is_processing = Column(Boolean, default=False)
    current_ticket_id = Column(String(100))
    
    # Performance metrics
    total_processed = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_count = Column(Integer, default=0)
    average_processing_time = Column(Float, default=0.0)
    
    last_activity = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AgentStatus(agent='{self.agent_name}', status='{self.status}')>"

class CollaborationMetrics(Base):
    """Model for tracking authentic agent collaboration and consensus building."""
    __tablename__ = 'collaboration_metrics'
    
    id = Column(Integer, primary_key=True)
    ticket_id = Column(String(100), nullable=False)
    
    # Initial disagreement tracking
    initial_disagreements = Column(JSON)  # {field: [agent_values], conflict_type: description}
    disagreement_count = Column(Integer, default=0)
    
    # Conflict resolution details
    conflicts_identified = Column(JSON)  # List of specific conflicts found
    conflict_resolution_methods = Column(JSON)  # How each conflict was resolved
    resolution_iterations = Column(Integer, default=0)  # Number of back-and-forth cycles
    
    # Consensus building process
    consensus_start_time = Column(DateTime)
    consensus_end_time = Column(DateTime)
    consensus_building_duration = Column(Float, default=0.0)  # in seconds
    
    # Final agreement tracking
    final_agreement_scores = Column(JSON)  # {field: agreement_score}
    overall_agreement_strength = Column(Float, default=0.0)
    consensus_reached = Column(Boolean, default=False)
    
    # Agent participation
    agent_iterations = Column(JSON)  # {agent_name: number_of_revisions}
    agent_agreement_evolution = Column(JSON)  # Timeline of how agreement evolved
    
    # Quality of consensus
    confidence_improvement = Column(Float, default=0.0)  # Initial vs final confidence
    result_stability = Column(Float, default=0.0)  # Likelihood result won't change
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<CollaborationMetrics(ticket_id='{self.ticket_id}', consensus={self.consensus_reached})>"

class ExperimentConfiguration(Base):
    """Model for storing experimental sweep configurations."""
    __tablename__ = 'experiment_configurations'
    
    id = Column(Integer, primary_key=True)
    experiment_name = Column(String(200), nullable=False)
    experiment_type = Column(String(100), nullable=False)  # 'model_assignment', 'agent_ordering', 'consensus_mechanism', etc.
    
    # Experiment parameters
    configuration = Column(JSON, nullable=False)  # Full config details
    description = Column(Text)
    
    # Status tracking
    status = Column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed'
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Results summary
    total_tickets_tested = Column(Integer, default=0)
    successful_runs = Column(Integer, default=0)
    failed_runs = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<ExperimentConfiguration(name='{self.experiment_name}', type='{self.experiment_type}')>"

class ExperimentRun(Base):
    """Model for tracking individual experiment executions."""
    __tablename__ = 'experiment_runs'
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiment_configurations.id'), nullable=False)
    run_number = Column(Integer, nullable=False)  # Sequential run number within experiment
    
    # Test data
    test_tickets = Column(JSON, nullable=False)  # List of tickets used for testing
    ticket_count = Column(Integer, nullable=False)
    
    # Execution details
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    execution_time = Column(Float)  # Total time in seconds
    
    # Status and results
    status = Column(String(50), default='running')  # 'running', 'completed', 'failed'
    error_message = Column(Text)
    
    # Performance metrics
    average_processing_time = Column(Float)
    total_processing_time = Column(Float)
    success_rate = Column(Float)  # Percentage of runs that completed without errors (technical completion)
    quality_success_rate = Column(Float)  # Percentage of runs with good quality results (accuracy > 0.7)
    
    # Quality metrics aggregates
    average_accuracy = Column(Float)
    average_relevancy = Column(Float) 
    average_faithfulness = Column(Float)
    average_hallucination = Column(Float)
    
    # Collaboration metrics aggregates
    average_consensus_time = Column(Float)
    average_agreement_strength = Column(Float)
    consensus_success_rate = Column(Float)
    average_resolution_iterations = Column(Float)
    
    # Raw results storage
    detailed_results = Column(JSON)  # Complete results for analysis
    
    # Relationships
    experiment = relationship("ExperimentConfiguration")
    
    def __repr__(self):
        return f"<ExperimentRun(experiment_id={self.experiment_id}, run={self.run_number})>"

class ExperimentResult(Base):
    """Model for storing individual ticket results within experiment runs."""
    __tablename__ = 'experiment_results'
    
    id = Column(Integer, primary_key=True)
    experiment_run_id = Column(Integer, ForeignKey('experiment_runs.id'), nullable=False)
    ticket_id = Column(String(100), nullable=False)
    
    # Configuration used for this specific result
    agent_configuration = Column(JSON)  # Model assignments, ordering, etc.
    
    # Processing results
    processing_time = Column(Float)
    processing_status = Column(String(50))
    
    # Quality metrics
    accuracy_score = Column(Float)
    relevancy_score = Column(Float)
    faithfulness_score = Column(Float)
    hallucination_score = Column(Float)
    overall_quality_score = Column(Float)
    
    # Classification results
    predicted_intent = Column(String(100))
    predicted_severity = Column(String(50))
    classification_confidence = Column(Float)
    
    # Collaboration metrics
    consensus_reached = Column(Boolean, default=False)
    consensus_time = Column(Float)
    agreement_strength = Column(Float)
    resolution_iterations = Column(Integer, default=0)
    
    # Raw outputs
    agent_outputs = Column(JSON)  # Individual agent outputs
    final_result = Column(JSON)   # Final collaborative result
    error_details = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment_run = relationship("ExperimentRun")
    
    def __repr__(self):
        return f"<ExperimentResult(run_id={self.experiment_run_id}, ticket='{self.ticket_id}')>"

class ExperimentComparison(Base):
    """Model for storing experiment comparison analyses."""
    __tablename__ = 'experiment_comparisons'
    
    id = Column(Integer, primary_key=True)
    comparison_name = Column(String(200), nullable=False)
    experiment_ids = Column(JSON, nullable=False)  # List of experiment IDs being compared
    
    # Comparison metrics
    comparison_type = Column(String(100))  # 'model_performance', 'agent_ordering', 'consensus_effectiveness'
    metric_focus = Column(String(100))     # 'accuracy', 'speed', 'consensus_quality', 'overall'
    
    # Results
    winner_experiment_id = Column(Integer)
    performance_rankings = Column(JSON)  # Ranked list of experiments
    statistical_significance = Column(JSON)  # P-values and confidence intervals
    
    # Analysis details
    comparison_results = Column(JSON)  # Detailed comparison data
    recommendations = Column(Text)     # Human-readable recommendations
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ExperimentComparison(name='{self.comparison_name}', experiments={len(self.experiment_ids or [])})>"

# Database setup and session management
def get_database_url():
    """Get the database URL from environment variables."""
    return os.environ.get('DATABASE_URL')

def create_db_engine():
    """Create database engine."""
    database_url = get_database_url()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(database_url, pool_pre_ping=True, pool_recycle=300)
    return engine

def get_db_session():
    """Get database session."""
    engine = create_db_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_database():
    """Initialize database tables."""
    engine = create_db_engine()
    Base.metadata.create_all(engine)
    
    # Initialize default agent statuses
    session = get_db_session()
    try:
        agents = ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']
        for agent_name in agents:
            existing = session.query(AgentStatus).filter_by(agent_name=agent_name).first()
            if not existing:
                agent_status = AgentStatus(agent_name=agent_name)
                session.add(agent_status)
        
        session.commit()
        # Only print once using a global flag to prevent duplicate import messages
        if not globals().get('_database_init_printed', False):
            print("Database initialized successfully")
            globals()['_database_init_printed'] = True
    except Exception as e:
        session.rollback()
        print(f"Error initializing database: {e}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    init_database()
