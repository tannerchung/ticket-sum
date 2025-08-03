"""
Database service layer for the Support Ticket Summarizer system.
Handles all database operations and integrations.
"""

from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor
from models import (
    SupportTicket, ProcessingLog, QualityEvaluation, AgentStatus, CollaborationMetrics,
    get_db_session, init_database
)

class DatabaseService:
    """Service class for database operations."""
    
    def __init__(self):
        """Initialize database service with connection pooling support."""
        try:
            init_database()
        except Exception as e:
            print(f"Warning: Database initialization failed: {e}")
        
        # Thread lock for safe concurrent operations
        self._lock = threading.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="db_worker")
    
    def save_ticket_result(self, result: Dict[str, Any]) -> bool:
        """Save a processed ticket result to the database."""
        session = get_db_session()
        try:
            ticket_id = result.get('ticket_id')
            
            # Check if ticket already exists
            existing_ticket = session.query(SupportTicket).filter_by(ticket_id=ticket_id).first()
            
            # Ensure required fields have default values to prevent null violations
            original_message = str(result.get('original_message') or 
                                  result.get('original_content') or 
                                  'No content available')
            
            if existing_ticket:
                # Update existing ticket
                existing_ticket.original_message = original_message  # type: ignore
                existing_ticket.intent = str(result.get('classification', {}).get('intent') or 'general_inquiry')  # type: ignore
                existing_ticket.severity = str(result.get('classification', {}).get('severity') or 'medium')  # type: ignore
                existing_ticket.classification_confidence = float(result.get('classification', {}).get('confidence') or 0.5)  # type: ignore
                existing_ticket.summary = str(result.get('summary') or 'Summary not available')  # type: ignore
                existing_ticket.action_recommendation = result.get('action_recommendation') or {}  # type: ignore
                existing_ticket.processing_status = str(result.get('processing_status', 'completed'))  # type: ignore
                existing_ticket.processed_at = datetime.now(timezone.utc)  # type: ignore
            else:
                # Create new ticket
                existing_ticket = SupportTicket(
                    ticket_id=str(ticket_id),
                    original_message=original_message,
                    intent=str(result.get('classification', {}).get('intent') or 'general_inquiry'),
                    severity=str(result.get('classification', {}).get('severity') or 'medium'),
                    classification_confidence=float(result.get('classification', {}).get('confidence') or 0.5),
                    summary=str(result.get('summary') or 'Summary not available'),
                    action_recommendation=result.get('action_recommendation') or {},
                    processing_status=str(result.get('processing_status', 'completed')),
                    processed_at=datetime.now(timezone.utc),
                    source='manual'
                )
                session.add(existing_ticket)
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving ticket result: {e}")
            return False
        finally:
            session.close()
    
    def save_ticket_results_bulk(self, results: List[Dict[str, Any]]) -> int:
        """
        Save multiple ticket results in a single transaction for better performance.
        
        Args:
            results: List of ticket result dictionaries
            
        Returns:
            Number of successfully saved tickets
        """
        if not results:
            return 0
        
        session = get_db_session()
        saved_count = 0
        
        try:
            with self._lock:  # Ensure thread safety
                for result in results:
                    if result.get('processing_status') == 'error':
                        continue  # Skip error results
                    
                    try:
                        ticket_id = result.get('ticket_id')
                        
                        # Check if ticket already exists
                        existing_ticket = session.query(SupportTicket).filter_by(ticket_id=ticket_id).first()
                        
                        # Ensure required fields have default values
                        original_message = str(result.get('original_message') or 
                                              result.get('original_content') or 
                                              'No content available')
                        
                        if existing_ticket:
                            # Update existing ticket
                            existing_ticket.original_message = original_message  # type: ignore
                            existing_ticket.intent = str(result.get('classification', {}).get('intent') or 'general_inquiry')  # type: ignore
                            existing_ticket.severity = str(result.get('classification', {}).get('severity') or 'medium')  # type: ignore
                            existing_ticket.classification_confidence = float(result.get('classification', {}).get('confidence') or 0.5)  # type: ignore
                            existing_ticket.summary = str(result.get('summary') or 'Summary not available')  # type: ignore
                            existing_ticket.action_recommendation = result.get('action_recommendation') or {}  # type: ignore
                            existing_ticket.processing_status = str(result.get('processing_status', 'completed'))  # type: ignore
                            existing_ticket.processed_at = datetime.now(timezone.utc)  # type: ignore
                        else:
                            # Create new ticket
                            new_ticket = SupportTicket(
                                ticket_id=str(ticket_id),
                                original_message=original_message,
                                intent=str(result.get('classification', {}).get('intent') or 'general_inquiry'),
                                severity=str(result.get('classification', {}).get('severity') or 'medium'),
                                classification_confidence=float(result.get('classification', {}).get('confidence') or 0.5),
                                summary=str(result.get('summary') or 'Summary not available'),
                                action_recommendation=result.get('action_recommendation') or {},
                                processing_status=str(result.get('processing_status', 'completed')),
                                processed_at=datetime.now(timezone.utc),
                                source='batch'
                            )
                            session.add(new_ticket)
                        
                        saved_count += 1
                        
                    except Exception as e:
                        print(f"Error saving ticket {result.get('ticket_id', 'unknown')} in bulk: {e}")
                        continue
                
                # Commit all changes at once
                session.commit()
                print(f"✅ Bulk saved {saved_count} tickets to database")
                
        except Exception as e:
            session.rollback()
            print(f"Error in bulk ticket save: {e}")
            saved_count = 0
        finally:
            session.close()
        
        return saved_count
    
    def save_evaluation_results_bulk(self, ticket_evaluations: List[Dict[str, Any]]) -> int:
        """
        Save multiple evaluation results in bulk for better performance.
        
        Args:
            ticket_evaluations: List of dicts with 'ticket_id' and 'evaluation_scores'
            
        Returns:
            Number of successfully saved evaluations
        """
        if not ticket_evaluations:
            return 0
        
        session = get_db_session()
        saved_count = 0
        
        try:
            with self._lock:
                evaluation_objects = []
                
                for item in ticket_evaluations:
                    ticket_id = item.get('ticket_id')
                    evaluation_scores = item.get('evaluation_scores', {})
                    
                    if not ticket_id or not evaluation_scores:
                        continue
                    
                    evaluation = QualityEvaluation(
                        ticket_id=ticket_id,
                        evaluation_scores=evaluation_scores,
                        evaluation_timestamp=datetime.now(timezone.utc),
                        evaluator_type='deepeval_parallel'
                    )
                    evaluation_objects.append(evaluation)
                    saved_count += 1
                
                # Bulk insert all evaluations
                if evaluation_objects:
                    session.add_all(evaluation_objects)
                    session.commit()
                    print(f"✅ Bulk saved {saved_count} evaluations to database")
                
        except Exception as e:
            session.rollback()
            print(f"Error in bulk evaluation save: {e}")
            saved_count = 0
        finally:
            session.close()
        
        return saved_count
    
    def save_processing_log(self, ticket_id: str, agent_name: str, 
                          input_data: Dict, output_data: Dict, 
                          metadata: Dict, status: str = 'success',
                          processing_time: float = 0.0, 
                          error_message: Optional[str] = None,
                          trace_id: Optional[str] = None,
                          langfuse_trace_id: Optional[str] = None,
                          langfuse_session_id: Optional[str] = None,
                          langfuse_observation_id: Optional[str] = None) -> bool:
        """Save a processing log entry."""
        session = get_db_session()
        try:
            safe_input_data = self._make_json_safe(input_data)
            safe_output_data = self._make_json_safe(output_data)
            safe_metadata = self._make_json_safe(metadata)
            
            log = ProcessingLog(
                ticket_id=ticket_id,
                agent_name=agent_name,
                input_data=safe_input_data,
                output_data=safe_output_data,
                processing_metadata=safe_metadata,
                processing_time=processing_time,
                status=status,
                error_message=error_message,
                trace_id=trace_id,
                langfuse_trace_id=langfuse_trace_id,
                langfuse_session_id=langfuse_session_id,
                langfuse_observation_id=langfuse_observation_id
            )
            
            session.add(log)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving processing log: {e}")
            return False
        finally:
            session.close()

    def save_processing_log_with_agent_stats(self, ticket_id: str, agent_name: str, 
                                           input_data: Dict, output_data: Dict, 
                                           metadata: Dict, status: str = 'success',
                                           processing_time: float = 0.0, 
                                           error_message: Optional[str] = None,
                                           trace_id: Optional[str] = None,
                                           langfuse_trace_id: Optional[str] = None,
                                           langfuse_session_id: Optional[str] = None,
                                           langfuse_observation_id: Optional[str] = None) -> bool:
        """Save processing log AND update agent statistics."""
        session = get_db_session()
        try:
            # Save the log (existing code)
            safe_input_data = self._make_json_safe(input_data)
            safe_output_data = self._make_json_safe(output_data)
            safe_metadata = self._make_json_safe(metadata)
            
            log = ProcessingLog(
                ticket_id=ticket_id,
                agent_name=agent_name,
                input_data=safe_input_data,
                output_data=safe_output_data,
                processing_metadata=safe_metadata,
                processing_time=processing_time,
                status=status,
                error_message=error_message,
                trace_id=trace_id,
                langfuse_trace_id=langfuse_trace_id,
                langfuse_session_id=langfuse_session_id,
                langfuse_observation_id=langfuse_observation_id
            )
            session.add(log)
            
            # UPDATE AGENT STATISTICS (this was missing!)
            agent_status = session.query(AgentStatus).filter_by(agent_name=agent_name).first()
            if not agent_status:
                agent_status = AgentStatus(agent_name=agent_name)
                session.add(agent_status)
            
            # Update statistics
            agent_status.total_processed = (agent_status.total_processed or 0) + 1
            agent_status.last_activity = datetime.now(timezone.utc)
            
            if status == 'success':
                agent_status.success_count = (agent_status.success_count or 0) + 1
            else:
                agent_status.error_count = (agent_status.error_count or 0) + 1
            
            # Update average processing time
            if agent_status.total_processed > 1:
                current_avg = agent_status.average_processing_time or 0
                new_avg = ((current_avg * (agent_status.total_processed - 1)) + processing_time) / agent_status.total_processed
                agent_status.average_processing_time = new_avg
            else:
                agent_status.average_processing_time = processing_time
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving processing log with stats: {e}")
            return False
        finally:
            session.close()

    def _make_json_safe(self, obj):
        """Convert objects to JSON-serializable format."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        if hasattr(obj, '__dict__'):
            return str(obj)
        if hasattr(obj, 'raw'):
            return str(obj.raw)
        return obj
    
    def save_quality_evaluation(self, ticket_id: str, evaluation_scores: Dict[str, float]) -> bool:
        """Save quality evaluation scores."""
        session = get_db_session()
        try:
            evaluation = QualityEvaluation(
                ticket_id=ticket_id,
                hallucination_score=evaluation_scores.get('hallucination', 0.0),
                relevancy_score=evaluation_scores.get('relevancy', 0.0),
                faithfulness_score=evaluation_scores.get('faithfulness', 0.0),
                overall_accuracy=evaluation_scores.get('overall_accuracy', 0.0),
                evaluation_model='deepeval'
            )
            
            session.add(evaluation)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving quality evaluation: {e}")
            return False
        finally:
            session.close()
    
    def update_agent_status(self, agent_name: str, status: str, 
                           is_processing: bool = False, 
                           current_ticket_id: Optional[str] = None) -> bool:
        """Update agent status in the database."""
        session = get_db_session()
        try:
            agent = session.query(AgentStatus).filter_by(agent_name=agent_name).first()
            if agent:
                agent.status = status  # type: ignore
                agent.is_processing = is_processing  # type: ignore
                agent.current_ticket_id = current_ticket_id  # type: ignore
                agent.last_activity = datetime.now(timezone.utc)  # type: ignore
                agent.updated_at = datetime.now(timezone.utc)  # type: ignore
            else:
                agent = AgentStatus(
                    agent_name=agent_name,
                    status=status,
                    is_processing=is_processing,
                    current_ticket_id=current_ticket_id
                )
                session.add(agent)
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error updating agent status: {e}")
            return False
        finally:
            session.close()
    
    def get_agent_statistics(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for an agent."""
        session = get_db_session()
        try:
            agent = session.query(AgentStatus).filter_by(agent_name=agent_name).first()
            if not agent:
                return {}
            
            return {
                'total_processed': agent.total_processed,
                'success_count': agent.success_count,
                'error_count': agent.error_count,
                'success_rate': agent.success_count / max(agent.total_processed or 0, 1) * 100,  # type: ignore
                'average_processing_time': agent.average_processing_time,
                'last_activity': agent.last_activity.isoformat() if agent.last_activity else None  # type: ignore
            }
            
        except Exception as e:
            print(f"Error getting agent statistics: {e}")
            return {}
        finally:
            session.close()

    def get_all_agent_statistics(self) -> List[Dict[str, Any]]:
        """Get performance statistics for all agents."""
        session = get_db_session()
        try:
            agents = session.query(AgentStatus).all()
            results = []
            
            for agent in agents:
                total_processed = agent.total_processed or 0
                success_count = agent.success_count or 0
                error_count = agent.error_count or 0
                
                results.append({
                    'agent_name': agent.agent_name,
                    'total_processed': total_processed,
                    'success_count': success_count,
                    'error_count': error_count,
                    'success_rate': (success_count / max(total_processed, 1)) * 100,
                    'average_processing_time': agent.average_processing_time or 0.0,
                    'last_activity': agent.last_activity.isoformat() if agent.last_activity else None,
                    'status': agent.status,
                    'is_processing': agent.is_processing,
                    'current_ticket_id': agent.current_ticket_id
                })
            
            return sorted(results, key=lambda x: x['total_processed'], reverse=True)
            
        except Exception as e:
            print(f"Error getting all agent statistics: {e}")
            return []
        finally:
            session.close()
    
    def get_recent_tickets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processed tickets."""
        session = get_db_session()
        try:
            tickets = session.query(SupportTicket)\
                           .order_by(SupportTicket.created_at.desc())\
                           .limit(limit)\
                           .all()
            
            result = []
            for ticket in tickets:
                result.append({
                    'ticket_id': ticket.ticket_id,
                    'intent': ticket.intent,
                    'severity': ticket.severity,
                    'summary': ticket.summary[:100] + '...' if ticket.summary and len(ticket.summary) > 100 else ticket.summary,  # type: ignore
                    'status': ticket.processing_status,
                    'created_at': ticket.created_at.isoformat() if ticket.created_at else None,  # type: ignore
                    'processed_at': ticket.processed_at.isoformat() if ticket.processed_at else None  # type: ignore
                })
            
            return result
            
        except Exception as e:
            print(f"Error getting recent tickets: {e}")
            return []
        finally:
            session.close()
    
    def get_processing_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get processing analytics for the last N days."""
        session = get_db_session()
        try:
            from sqlalchemy import func
            from datetime import timedelta
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # Basic counts
            total_tickets = session.query(SupportTicket).filter(
                SupportTicket.created_at >= cutoff_date
            ).count()
            
            # Intent distribution
            intent_stats = session.query(
                SupportTicket.intent,
                func.count(SupportTicket.intent).label('count')
            ).filter(
                SupportTicket.created_at >= cutoff_date
            ).group_by(SupportTicket.intent).all()
            
            # Severity distribution
            severity_stats = session.query(
                SupportTicket.severity,
                func.count(SupportTicket.severity).label('count')
            ).filter(
                SupportTicket.created_at >= cutoff_date
            ).group_by(SupportTicket.severity).all()
            
            # Quality metrics average
            quality_avg = session.query(
                func.avg(QualityEvaluation.hallucination_score).label('avg_hallucination'),
                func.avg(QualityEvaluation.relevancy_score).label('avg_relevancy'),
                func.avg(QualityEvaluation.faithfulness_score).label('avg_faithfulness'),
                func.avg(QualityEvaluation.overall_accuracy).label('avg_accuracy')
            ).filter(
                QualityEvaluation.created_at >= cutoff_date
            ).first()
            
            return {
                'total_tickets': total_tickets,
                'intent_distribution': {stat.intent or 'unknown': stat.count for stat in intent_stats},
                'severity_distribution': {stat.severity or 'unknown': stat.count for stat in severity_stats},
                'quality_metrics': {
                    'hallucination': float(quality_avg.avg_hallucination or 0),
                    'relevancy': float(quality_avg.avg_relevancy or 0),
                    'faithfulness': float(quality_avg.avg_faithfulness or 0),
                    'accuracy': float(quality_avg.avg_accuracy or 0)
                } if quality_avg else {}
            }
            
        except Exception as e:
            print(f"Error getting processing analytics: {e}")
            return {}
        finally:
            session.close()
    
    def get_processing_logs(self, ticket_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get processing logs, optionally filtered by ticket ID."""
        session = get_db_session()
        try:
            query = session.query(ProcessingLog)
            
            if ticket_id:
                query = query.filter(ProcessingLog.ticket_id == ticket_id)
            
            logs = query.order_by(ProcessingLog.created_at.desc()).limit(limit).all()
            
            result = []
            for log in logs:
                result.append({
                    'ticket_id': log.ticket_id,
                    'agent_name': log.agent_name,
                    'status': log.status,
                    'processing_time': log.processing_time,
                    'created_at': log.created_at.isoformat() if log.created_at else None,  # type: ignore
                    'trace_id': log.trace_id,
                    'error_message': log.error_message
                })
            
            return result
            
        except Exception as e:
            print(f"Error getting processing logs: {e}")
            return []
        finally:
            session.close()

    def save_collaboration_metrics(self, ticket_id: str, metrics: Dict[str, Any]) -> bool:
        """Save authentic collaboration metrics to the database."""
        session = get_db_session()
        try:
            collaboration = CollaborationMetrics(
                ticket_id=ticket_id,
                initial_disagreements=metrics.get('initial_disagreements', {}),
                disagreement_count=metrics.get('disagreement_count', 0),
                conflicts_identified=metrics.get('conflicts_identified', []),
                conflict_resolution_methods=metrics.get('conflict_resolution_methods', []),
                resolution_iterations=metrics.get('resolution_iterations', 0),
                consensus_start_time=datetime.now(timezone.utc),
                consensus_end_time=datetime.now(timezone.utc),
                consensus_building_duration=metrics.get('consensus_building_duration', 0.0),
                final_agreement_scores=metrics.get('final_agreement_scores', {}),
                overall_agreement_strength=metrics.get('overall_agreement_strength', 0.0),
                consensus_reached=metrics.get('consensus_reached', False),
                agent_iterations=metrics.get('agent_iterations', {}),
                agent_agreement_evolution=metrics.get('agent_agreement_evolution', []),
                confidence_improvement=metrics.get('confidence_improvement', 0.0),
                result_stability=metrics.get('result_stability', 0.0)
            )
            
            session.add(collaboration)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error saving collaboration metrics: {e}")
            return False
        finally:
            session.close()

    def get_collaboration_analytics(self) -> Dict[str, Any]:
        """Get analytics on agent collaboration and consensus building."""
        session = get_db_session()
        try:
            metrics = session.query(CollaborationMetrics).all()
            
            if not metrics:
                return {"total_collaborations": 0}
            
            total_collaborations = len(metrics)
            consensus_achieved = sum(1 for m in metrics if m.consensus_reached)  # type: ignore
            avg_disagreements = sum(m.disagreement_count for m in metrics) / total_collaborations
            avg_consensus_time = sum(m.consensus_building_duration for m in metrics) / total_collaborations
            avg_agreement_strength = sum(m.overall_agreement_strength for m in metrics) / total_collaborations
            
            # Most common conflicts
            all_conflicts = []
            for m in metrics:
                if m.conflicts_identified:  # type: ignore
                    all_conflicts.extend(m.conflicts_identified)  # type: ignore
            
            conflict_frequency = {}
            for conflict in all_conflicts:
                conflict_type = conflict.split(':')[0] if ':' in conflict else conflict
                conflict_frequency[conflict_type] = conflict_frequency.get(conflict_type, 0) + 1
            
            return {
                "total_collaborations": total_collaborations,
                "consensus_success_rate": (consensus_achieved / total_collaborations) * 100,
                "avg_disagreements_per_ticket": avg_disagreements,
                "avg_consensus_building_time": avg_consensus_time,
                "avg_agreement_strength": avg_agreement_strength,
                "most_common_conflicts": sorted(conflict_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
                "total_conflicts_resolved": sum(len(m.conflict_resolution_methods or []) for m in metrics)  # type: ignore
            }
            
        except Exception as e:
            print(f"Error getting collaboration analytics: {e}")
            return {"error": str(e)}
        finally:
            session.close()


    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Get all experiment data from processing logs and experiment tables."""
        session = get_db_session()
        try:
            # Get processing logs with experiment metadata
            logs = session.query(ProcessingLog).filter(
                ProcessingLog.metadata != None
            ).order_by(ProcessingLog.created_at.desc()).all()
            
            # Filter for experiments in Python
            experiment_logs = []
            for log in logs:
                if log.metadata and 'experiment_id' in log.metadata:
                    experiment_logs.append(log)
            logs = experiment_logs
            
            experiments = []
            seen_experiments = set()
            
            for log in logs:
                metadata = log.metadata or {}
                experiment_id = metadata.get('experiment_id')
                
                if experiment_id and experiment_id not in seen_experiments:
                    seen_experiments.add(experiment_id)
                    
                    experiment_data = {
                        'experiment_id': experiment_id,
                        'experiment_type': metadata.get('experiment_type', 'unknown'),
                        'model_name': metadata.get('model_name', 'unknown'),
                        'accuracy': float(metadata.get('accuracy', 0.0)),
                        'processing_time': float(log.processing_time or 0.0),
                        'status': log.status or 'unknown',
                        'created_at': log.created_at,
                        'temperature': float(metadata.get('temperature', 0.1)),
                        'ticket_count': int(metadata.get('ticket_count', 1)),
                        'success_rate': float(metadata.get('success_rate', 0.0)),
                        'quality_scores': metadata.get('quality_scores', {}),
                        'agent_name': log.agent_name
                    }
                    experiments.append(experiment_data)
            
            return experiments
            
        except Exception as e:
            print(f"Error getting experiments: {e}")
            return []
        finally:
            session.close()

    def get_experiment_configuration_analysis(self) -> Dict[str, Any]:
        """Get analysis of which experiment configurations perform best."""
        session = get_db_session()
        try:
            # Get all processing logs with experiment metadata
            logs = session.query(ProcessingLog).filter(
                ProcessingLog.metadata != None
            ).all()
            
            # Analyze different configuration aspects
            model_performance = {}
            agent_order_performance = {}
            consensus_performance = {}
            quality_threshold_performance = {}
            
            for log in logs:
                metadata = log.metadata or {}
                
                # Skip if not an experiment
                if not metadata.get('experiment_id'):
                    continue
                
                # Extract configuration details and performance metrics
                accuracy = float(metadata.get('accuracy', 0.0))
                processing_time = float(log.processing_time or 0.0)
                success = log.status == 'success'
                
                # Model analysis
                model_name = metadata.get('model_name', 'unknown')
                if model_name not in model_performance:
                    model_performance[model_name] = {
                        'accuracies': [], 'times': [], 'successes': 0, 'total': 0
                    }
                model_performance[model_name]['accuracies'].append(accuracy)
                model_performance[model_name]['times'].append(processing_time)
                model_performance[model_name]['successes'] += 1 if success else 0
                model_performance[model_name]['total'] += 1
                
                # Agent order analysis
                agent_order = metadata.get('agent_order', 'standard')
                if agent_order not in agent_order_performance:
                    agent_order_performance[agent_order] = {
                        'accuracies': [], 'times': [], 'successes': 0, 'total': 0
                    }
                agent_order_performance[agent_order]['accuracies'].append(accuracy)
                agent_order_performance[agent_order]['times'].append(processing_time)
                agent_order_performance[agent_order]['successes'] += 1 if success else 0
                agent_order_performance[agent_order]['total'] += 1
                
                # Consensus mechanism analysis
                consensus_method = metadata.get('consensus_mechanism', 'majority_vote')
                if consensus_method not in consensus_performance:
                    consensus_performance[consensus_method] = {
                        'accuracies': [], 'times': [], 'successes': 0, 'total': 0
                    }
                consensus_performance[consensus_method]['accuracies'].append(accuracy)
                consensus_performance[consensus_method]['times'].append(processing_time)
                consensus_performance[consensus_method]['successes'] += 1 if success else 0
                consensus_performance[consensus_method]['total'] += 1
                
                # Quality threshold analysis
                quality_threshold = metadata.get('quality_threshold', 0.8)
                threshold_key = f"{quality_threshold:.1f}"
                if threshold_key not in quality_threshold_performance:
                    quality_threshold_performance[threshold_key] = {
                        'accuracies': [], 'times': [], 'successes': 0, 'total': 0
                    }
                quality_threshold_performance[threshold_key]['accuracies'].append(accuracy)
                quality_threshold_performance[threshold_key]['times'].append(processing_time)
                quality_threshold_performance[threshold_key]['successes'] += 1 if success else 0
                quality_threshold_performance[threshold_key]['total'] += 1
            
            # Calculate summary statistics for each category
            def calculate_stats(performance_dict):
                results = {}
                for config, data in performance_dict.items():
                    if data['total'] > 0:
                        results[config] = {
                            'avg_accuracy': sum(data['accuracies']) / len(data['accuracies']) if data['accuracies'] else 0,
                            'avg_time': sum(data['times']) / len(data['times']) if data['times'] else 0,
                            'success_rate': data['successes'] / data['total'],
                            'total_experiments': data['total'],
                            'efficiency_score': (sum(data['accuracies']) / len(data['accuracies'])) / (sum(data['times']) / len(data['times'])) if data['times'] and data['accuracies'] else 0
                        }
                return results
            
            model_stats = calculate_stats(model_performance)
            agent_order_stats = calculate_stats(agent_order_performance)
            consensus_stats = calculate_stats(consensus_performance)
            quality_threshold_stats = calculate_stats(quality_threshold_performance)
            
            # Find winners
            def find_winner(stats_dict, metric='avg_accuracy'):
                if not stats_dict:
                    return None, 0
                best_config = max(stats_dict.items(), key=lambda x: x[1][metric])
                return best_config[0], best_config[1][metric]
            
            winners = {
                'best_model': find_winner(model_stats, 'avg_accuracy'),
                'fastest_model': find_winner(model_stats, 'efficiency_score'),
                'best_agent_order': find_winner(agent_order_stats, 'avg_accuracy'),
                'best_consensus': find_winner(consensus_stats, 'avg_accuracy'),
                'best_quality_threshold': find_winner(quality_threshold_stats, 'avg_accuracy')
            }
            
            return {
                'model_performance': model_stats,
                'agent_order_performance': agent_order_stats,
                'consensus_performance': consensus_stats,
                'quality_threshold_performance': quality_threshold_stats,
                'winners': winners,
                'total_experiments_analyzed': sum(data['total'] for data in model_performance.values())
            }
            
        except Exception as e:
            print(f"Error analyzing experiment configurations: {e}")
            return {}
        finally:
            session.close()

# Global database service instance
db_service = DatabaseService()
