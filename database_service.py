"""
Database service layer for the Support Ticket Summarizer system.
Handles all database operations and integrations.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from models import (
    SupportTicket, ProcessingLog, QualityEvaluation, AgentStatus, CollaborationMetrics,
    get_db_session, init_database
)

class DatabaseService:
    """Service class for database operations."""
    
    def __init__(self):
        """Initialize database service."""
        try:
            init_database()
        except Exception as e:
            print(f"Warning: Database initialization failed: {e}")
    
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
                existing_ticket.processed_at = datetime.utcnow()  # type: ignore
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
                    processed_at=datetime.utcnow(),
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
    
    def save_processing_log(self, ticket_id: str, agent_name: str, 
                          input_data: Dict, output_data: Dict, 
                          metadata: Dict, status: str = 'success',
                          processing_time: float = 0.0, 
                          error_message: Optional[str] = None,
                          trace_id: Optional[str] = None,
                          langsmith_run_id: Optional[str] = None) -> bool:
        """Save a processing log entry."""
        session = get_db_session()
        try:
            # Serialize data to ensure JSON compatibility
            def make_json_safe(obj):
                """Convert objects to JSON-serializable format."""
                if obj is None:
                    return {}
                if hasattr(obj, '__dict__'):
                    return str(obj)
                if hasattr(obj, 'raw'):
                    return str(obj.raw)
                return obj
            
            safe_input_data = make_json_safe(input_data)
            safe_output_data = make_json_safe(output_data)
            safe_metadata = make_json_safe(metadata)
            
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
                langsmith_run_id=langsmith_run_id
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
                agent.last_activity = datetime.utcnow()  # type: ignore
                agent.updated_at = datetime.utcnow()  # type: ignore
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
                'success_rate': agent.success_count / max(agent.total_processed, 1) * 100,
                'average_processing_time': agent.average_processing_time,
                'last_activity': agent.last_activity.isoformat() if agent.last_activity else None
            }
            
        except Exception as e:
            print(f"Error getting agent statistics: {e}")
            return {}
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
                    'summary': ticket.summary[:100] + '...' if ticket.summary and len(ticket.summary) > 100 else ticket.summary,
                    'status': ticket.processing_status,
                    'created_at': ticket.created_at.isoformat() if ticket.created_at else None,
                    'processed_at': ticket.processed_at.isoformat() if ticket.processed_at else None
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
            from sqlalchemy import func, and_
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
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
                    'created_at': log.created_at.isoformat() if log.created_at else None,
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
                consensus_start_time=datetime.utcnow(),
                consensus_end_time=datetime.utcnow(),
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
            consensus_achieved = sum(1 for m in metrics if m.consensus_reached)
            avg_disagreements = sum(m.disagreement_count for m in metrics) / total_collaborations
            avg_consensus_time = sum(m.consensus_building_duration for m in metrics) / total_collaborations
            avg_agreement_strength = sum(m.overall_agreement_strength for m in metrics) / total_collaborations
            
            # Most common conflicts
            all_conflicts = []
            for m in metrics:
                if m.conflicts_identified:
                    all_conflicts.extend(m.conflicts_identified)
            
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
                "total_conflicts_resolved": sum(len(m.conflict_resolution_methods or []) for m in metrics)
            }
            
        except Exception as e:
            print(f"Error getting collaboration analytics: {e}")
            return {"error": str(e)}
        finally:
            session.close()


# Global database service instance
db_service = DatabaseService()
