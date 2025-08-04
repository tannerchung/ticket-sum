"""
Experimental Sweep Framework for Multi-Agent Optimization
Allows systematic testing of different agent configurations to find optimal setups.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import itertools
import numpy as np
from collections import defaultdict

from agents import CollaborativeSupportCrew
from models import (
    ExperimentConfiguration, ExperimentRun, ExperimentResult, ExperimentComparison,
    get_db_session
)
from config import AVAILABLE_MODELS


class ExperimentType(Enum):
    """Types of experiments that can be run."""
    MODEL_ASSIGNMENT = "model_assignment"
    AGENT_ORDERING = "agent_ordering" 
    CONSENSUS_MECHANISM = "consensus_mechanism"
    PROMPT_ENGINEERING = "prompt_engineering"
    QUALITY_THRESHOLD = "quality_threshold"
    COMPREHENSIVE_SWEEP = "comprehensive_sweep"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    experiment_type: ExperimentType
    description: str
    
    # Model assignment configurations
    model_assignments: Optional[Dict[str, List[str]]] = None  # {agent: [models_to_test]}
    
    # Agent ordering configurations
    agent_orderings: Optional[List[List[str]]] = None  # Different ordering sequences
    
    # Consensus mechanism configurations
    consensus_mechanisms: Optional[List[Dict[str, Any]]] = None
    
    # Prompt engineering configurations
    prompt_variations: Optional[Dict[str, List[str]]] = None  # {agent: [prompt_variations]}
    
    # Quality threshold configurations
    quality_thresholds: Optional[List[Dict[str, float]]] = None
    
    # Test data configuration
    test_tickets: List[Dict[str, str]] = None
    num_runs: int = 3  # Number of times to run each configuration
    
    # Parallel processing settings
    max_concurrent: int = 3  # Lower for experiments to avoid rate limiting


class ExperimentManager:
    """Manages experimental sweeps and analysis."""
    
    def __init__(self):
        self.session = get_db_session()
        
    def create_model_assignment_experiment(
        self, 
        name: str, 
        test_tickets: List[Dict[str, str]],
        agents_to_test: Optional[List[str]] = None,
        models_to_test: Optional[List[str]] = None
    ) -> ExperimentConfig:
        """
        Create an experiment to test different model assignments to agents.
        
        Args:
            name: Experiment name
            test_tickets: Tickets to test on
            agents_to_test: Which agents to vary (default: all)
            models_to_test: Which models to test (default: high-performing ones)
        """
        if agents_to_test is None:
            agents_to_test = ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"]
        
        if models_to_test is None:
            # Select high-performing models for testing
            models_to_test = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
        
        model_assignments = {}
        for agent in agents_to_test:
            model_assignments[agent] = models_to_test
        
        return ExperimentConfig(
            name=name,
            experiment_type=ExperimentType.MODEL_ASSIGNMENT,
            description=f"Testing {len(models_to_test)} models across {len(agents_to_test)} agents",
            model_assignments=model_assignments,
            test_tickets=test_tickets
        )
    
    def create_agent_ordering_experiment(
        self,
        name: str,
        test_tickets: List[Dict[str, str]],
        orderings_to_test: Optional[List[List[str]]] = None
    ) -> ExperimentConfig:
        """
        Create an experiment to test different agent execution orders.
        
        Args:
            name: Experiment name  
            test_tickets: Tickets to test on
            orderings_to_test: Different agent orderings to test
        """
        if orderings_to_test is None:
            # Generate some interesting orderings to test
            base_agents = ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"]
            orderings_to_test = [
                base_agents,  # Original order
                ["ticket_analyst", "triage_specialist", "support_strategist", "qa_reviewer"],  # Analysis first
                ["triage_specialist", "support_strategist", "ticket_analyst", "qa_reviewer"],  # Strategy before analysis
                ["qa_reviewer", "triage_specialist", "ticket_analyst", "support_strategist"],  # QA first (unusual)
            ]
        
        return ExperimentConfig(
            name=name,
            experiment_type=ExperimentType.AGENT_ORDERING,
            description=f"Testing {len(orderings_to_test)} different agent execution orders",
            agent_orderings=orderings_to_test,
            test_tickets=test_tickets
        )
    
    def create_consensus_mechanism_experiment(
        self,
        name: str,
        test_tickets: List[Dict[str, str]],
        mechanisms_to_test: Optional[List[Dict[str, Any]]] = None
    ) -> ExperimentConfig:
        """
        Create an experiment to test different consensus building mechanisms.
        
        Args:
            name: Experiment name
            test_tickets: Tickets to test on  
            mechanisms_to_test: Different consensus mechanisms to test
        """
        if mechanisms_to_test is None:
            mechanisms_to_test = [
                {"type": "majority_vote", "threshold": 0.5},
                {"type": "weighted_consensus", "weights": {"qa_reviewer": 0.4, "support_strategist": 0.3, "ticket_analyst": 0.2, "triage_specialist": 0.1}},
                {"type": "confidence_weighted", "min_confidence": 0.7},
                {"type": "iterative_refinement", "max_iterations": 3, "convergence_threshold": 0.8}
            ]
        
        return ExperimentConfig(
            name=name,
            experiment_type=ExperimentType.CONSENSUS_MECHANISM,
            description=f"Testing {len(mechanisms_to_test)} consensus building mechanisms", 
            consensus_mechanisms=mechanisms_to_test,
            test_tickets=test_tickets
        )
    
    def create_quality_threshold_experiment(
        self,
        name: str,
        test_tickets: List[Dict[str, str]],
        thresholds_to_test: Optional[List[Dict[str, float]]] = None
    ) -> ExperimentConfig:
        """
        Create an experiment to test different quality thresholds for re-processing.
        
        Args:
            name: Experiment name
            test_tickets: Tickets to test on
            thresholds_to_test: Different quality threshold configurations
        """
        if thresholds_to_test is None:
            thresholds_to_test = [
                {"faithfulness": 0.6, "relevancy": 0.7, "hallucination": 0.3},  # Lenient
                {"faithfulness": 0.7, "relevancy": 0.8, "hallucination": 0.2},  # Moderate
                {"faithfulness": 0.8, "relevancy": 0.9, "hallucination": 0.1},  # Strict
                {"faithfulness": 0.9, "relevancy": 0.95, "hallucination": 0.05}, # Very strict
            ]
        
        return ExperimentConfig(
            name=name,
            experiment_type=ExperimentType.QUALITY_THRESHOLD,
            description=f"Testing {len(thresholds_to_test)} quality threshold configurations",
            quality_thresholds=thresholds_to_test,
            test_tickets=test_tickets
        )
    
    def create_comprehensive_sweep(
        self,
        name: str,
        test_tickets: List[Dict[str, str]],
        limited_scope: bool = True
    ) -> ExperimentConfig:
        """
        Create a comprehensive experiment testing multiple dimensions.
        
        Args:
            name: Experiment name
            test_tickets: Tickets to test on
            limited_scope: If True, test fewer combinations to avoid explosion
        """
        if limited_scope:
            # Curated selection for manageable comprehensive testing
            model_assignments = {
                "triage_specialist": ["gpt-4o", "claude-3-5-haiku-20241022"],
                "ticket_analyst": ["gpt-4o", "claude-3-5-sonnet-20241022"], 
                "support_strategist": ["gpt-4o", "gpt-4o-mini"],
                "qa_reviewer": ["claude-3-5-sonnet-20241022", "gpt-4o"]
            }
            
            agent_orderings = [
                ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"],
                ["ticket_analyst", "triage_specialist", "support_strategist", "qa_reviewer"]
            ]
        else:
            # Full comprehensive testing (warning: can be very large)
            model_assignments = {
                agent: ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
                for agent in ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"]
            }
            
            agent_orderings = [
                ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"],
                ["ticket_analyst", "triage_specialist", "support_strategist", "qa_reviewer"],
                ["triage_specialist", "support_strategist", "ticket_analyst", "qa_reviewer"]
            ]
        
        return ExperimentConfig(
            name=name,
            experiment_type=ExperimentType.COMPREHENSIVE_SWEEP,
            description=f"Comprehensive multi-dimensional testing ({'limited' if limited_scope else 'full'} scope)",
            model_assignments=model_assignments,
            agent_orderings=agent_orderings,
            test_tickets=test_tickets,
            num_runs=2 if limited_scope else 1  # Fewer runs for comprehensive testing
        )
    
    async def run_experiment(self, config: ExperimentConfig) -> int:
        """
        Execute an experimental sweep.
        
        Args:
            config: Experiment configuration
            
        Returns:
            experiment_id: Database ID of the created experiment
        """
        print(f"\n🧪 Starting experiment: {config.name}")
        print(f"📊 Type: {config.experiment_type.value}")
        print(f"🎫 Test tickets: {len(config.test_tickets)}")
        
        # Save experiment configuration to database
        config_dict = asdict(config)
        # Convert enum to string for JSON serialization
        config_dict['experiment_type'] = config.experiment_type.value
        
        experiment = ExperimentConfiguration(
            experiment_name=config.name,
            experiment_type=config.experiment_type.value,
            configuration=config_dict,
            description=config.description,
            status='running',
            started_at=datetime.now(timezone.utc)
        )
        
        self.session.add(experiment)
        self.session.commit()
        experiment_id = experiment.id
        
        try:
            # Generate all configuration combinations to test
            configurations = self._generate_configurations(config)
            print(f"🔬 Generated {len(configurations)} configurations to test")
            
            total_runs = len(configurations) * config.num_runs
            print(f"⚡ Total runs: {total_runs}")
            
            run_number = 1
            for i, test_config in enumerate(configurations):
                print(f"\n🔧 Testing configuration {i+1}/{len(configurations)}")
                print(f"📋 Config: {self._format_config_summary(test_config)}")
                
                for run in range(config.num_runs):
                    print(f"🏃 Run {run+1}/{config.num_runs} (Overall: {run_number}/{total_runs})")
                    
                    await self._execute_single_run(
                        experiment_id, 
                        run_number, 
                        test_config, 
                        config.test_tickets,
                        config.max_concurrent
                    )
                    run_number += 1
            
            # Mark experiment as completed
            experiment.status = 'completed'
            experiment.completed_at = datetime.now(timezone.utc)
            
            # Calculate summary statistics
            summary_stats = self._calculate_experiment_summary(experiment_id)
            experiment.total_tickets_tested = summary_stats['total_tickets']
            experiment.successful_runs = summary_stats['successful_runs']
            experiment.failed_runs = summary_stats['failed_runs']
            
            self.session.commit()
            
            print(f"\n✅ Experiment '{config.name}' completed successfully!")
            print(f"📊 Results: {experiment.successful_runs} successful, {experiment.failed_runs} failed runs")
            
            return experiment_id
            
        except Exception as e:
            print(f"\n❌ Experiment failed: {str(e)}")
            experiment.status = 'failed'
            experiment.completed_at = datetime.now(timezone.utc)
            self.session.commit()
            raise
    
    def _generate_configurations(self, config: ExperimentConfig) -> List[Dict[str, Any]]:
        """Generate all configuration combinations to test."""
        configurations = []
        
        if config.experiment_type == ExperimentType.MODEL_ASSIGNMENT:
            # Generate all combinations of model assignments
            agent_names = list(config.model_assignments.keys())
            model_lists = [config.model_assignments[agent] for agent in agent_names]
            
            for model_combo in itertools.product(*model_lists):
                test_config = {
                    "type": "model_assignment",
                    "model_assignments": dict(zip(agent_names, model_combo))
                }
                configurations.append(test_config)
        
        elif config.experiment_type == ExperimentType.AGENT_ORDERING:
            for ordering in config.agent_orderings:
                test_config = {
                    "type": "agent_ordering", 
                    "agent_order": ordering
                }
                configurations.append(test_config)
        
        elif config.experiment_type == ExperimentType.CONSENSUS_MECHANISM:
            for mechanism in config.consensus_mechanisms:
                test_config = {
                    "type": "consensus_mechanism",
                    "consensus_config": mechanism
                }
                configurations.append(test_config)
        
        elif config.experiment_type == ExperimentType.QUALITY_THRESHOLD:
            for threshold_config in config.quality_thresholds:
                test_config = {
                    "type": "quality_threshold",
                    "thresholds": threshold_config
                }
                configurations.append(test_config)
        
        elif config.experiment_type == ExperimentType.COMPREHENSIVE_SWEEP:
            # Combine model assignments and agent orderings
            agent_names = list(config.model_assignments.keys())
            model_lists = [config.model_assignments[agent] for agent in agent_names]
            
            for model_combo in itertools.product(*model_lists):
                for ordering in config.agent_orderings:
                    test_config = {
                        "type": "comprehensive",
                        "model_assignments": dict(zip(agent_names, model_combo)),
                        "agent_order": ordering
                    }
                    configurations.append(test_config)
        
        return configurations
    
    def _format_config_summary(self, test_config: Dict[str, Any]) -> str:
        """Create a readable summary of a test configuration."""
        if test_config["type"] == "model_assignment":
            models = test_config["model_assignments"]
            return f"Models: {', '.join([f'{agent}={model}' for agent, model in models.items()])}"
        
        elif test_config["type"] == "agent_ordering":
            return f"Order: {' → '.join(test_config['agent_order'])}"
        
        elif test_config["type"] == "consensus_mechanism":
            mechanism = test_config["consensus_config"]
            return f"Consensus: {mechanism['type']}"
        
        elif test_config["type"] == "quality_threshold":
            thresholds = test_config["thresholds"]
            return f"Thresholds: {', '.join([f'{k}={v}' for k, v in thresholds.items()])}"
        
        elif test_config["type"] == "comprehensive":
            models = test_config["model_assignments"]
            order = test_config["agent_order"]
            model_summary = ', '.join([f'{agent}={model}' for agent, model in models.items()])
            return f"Models: {model_summary} | Order: {' → '.join(order)}"
        
        return str(test_config)
    
    async def _execute_single_run(
        self, 
        experiment_id: int, 
        run_number: int, 
        test_config: Dict[str, Any], 
        test_tickets: List[Dict[str, str]],
        max_concurrent: int
    ) -> None:
        """Execute a single experimental run."""
        start_time = time.time()
        
        # Create experiment run record
        experiment_run = ExperimentRun(
            experiment_id=experiment_id,
            run_number=run_number,
            test_tickets=test_tickets,
            ticket_count=len(test_tickets),
            started_at=datetime.now(timezone.utc)
        )
        
        self.session.add(experiment_run)
        self.session.commit()
        run_id = experiment_run.id
        
        try:
            # Create crew with experimental configuration
            crew = self._create_experimental_crew(test_config)
            
            # Process tickets (using parallel processing for efficiency)
            if len(test_tickets) > 1:
                results = await crew.process_tickets_parallel(test_tickets, max_concurrent)
            else:
                # Single ticket processing
                ticket = test_tickets[0] 
                result = crew.process_ticket_collaboratively(ticket["id"], ticket["content"])
                results = [result]
            
            # Save individual results and calculate aggregates
            aggregate_metrics = self._save_experiment_results(run_id, results, test_config)
            
            # Update run with aggregated metrics
            execution_time = time.time() - start_time
            experiment_run.completed_at = datetime.now(timezone.utc)
            experiment_run.execution_time = execution_time
            experiment_run.status = 'completed'
            
            # Set aggregate metrics
            experiment_run.success_rate = aggregate_metrics['completion_rate']  # Keep DB field name for compatibility
            experiment_run.quality_success_rate = aggregate_metrics['quality_success_rate']
            experiment_run.average_processing_time = aggregate_metrics['avg_processing_time']
            experiment_run.total_processing_time = aggregate_metrics['total_processing_time']
            experiment_run.average_accuracy = aggregate_metrics['avg_accuracy']
            experiment_run.average_relevancy = aggregate_metrics['avg_relevancy']
            experiment_run.average_faithfulness = aggregate_metrics['avg_faithfulness']
            experiment_run.average_hallucination = aggregate_metrics['avg_hallucination']
            experiment_run.average_consensus_time = aggregate_metrics['avg_consensus_time']
            experiment_run.average_agreement_strength = aggregate_metrics['avg_agreement_strength']
            experiment_run.detailed_results = [r for r in results if r]  # Store full results
            
            self.session.commit()
            
            print(f"✅ Run {run_number} completed in {execution_time:.2f}s")
            print(f"📊 Completion rate: {aggregate_metrics['completion_rate']:.1%}")
            print(f"🎯 Quality success rate: {aggregate_metrics['quality_success_rate']:.1%}")
            
        except Exception as e:
            print(f"❌ Run {run_number} failed: {str(e)}")
            experiment_run.status = 'failed'
            experiment_run.error_message = str(e)
            experiment_run.completed_at = datetime.now(timezone.utc)
            experiment_run.execution_time = time.time() - start_time
            self.session.commit()
    
    def _create_experimental_crew(self, test_config: Dict[str, Any]) -> CollaborativeSupportCrew:
        """Create a crew configured for experimental testing."""
        if test_config["type"] in ["model_assignment", "comprehensive"]:
            # Use custom model assignments
            model_assignments = test_config["model_assignments"]
            crew = CollaborativeSupportCrew(agent_models=model_assignments)
        else:
            # Use default model assignments
            crew = CollaborativeSupportCrew()
        
        # TODO: Implement agent ordering and consensus mechanism modifications
        # This would require extending the CollaborativeSupportCrew class
        
        return crew
    
    def _save_experiment_results(
        self, 
        run_id: int, 
        results: List[Dict[str, Any]], 
        test_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Save individual experiment results and return aggregate metrics."""
        successful_results = []
        
        for result in results:
            if not result or result.get('processing_status') == 'error':
                continue
            
            # Extract metrics from result
            classification = result.get('classification', {})
            
            # Save to experiment_results table
            experiment_result = ExperimentResult(
                experiment_run_id=run_id,
                ticket_id=result.get('ticket_id', 'unknown'),
                agent_configuration=test_config,
                processing_time=result.get('processing_time', 0.0),
                processing_status=result.get('processing_status', 'unknown'),
                predicted_intent=classification.get('intent'),
                predicted_severity=classification.get('severity'),
                classification_confidence=classification.get('confidence', 0.0),
                agent_outputs=result.get('individual_agent_logs', []),
                final_result=result
            )
            
            # Add collaboration metrics if available
            collaboration_metrics = result.get('collaboration_metrics', {})
            if collaboration_metrics:
                experiment_result.consensus_reached = collaboration_metrics.get('consensus_reached', False)
                experiment_result.consensus_time = collaboration_metrics.get('consensus_building_duration', 0.0)
                experiment_result.agreement_strength = collaboration_metrics.get('overall_agreement_strength', 0.0)
                experiment_result.resolution_iterations = collaboration_metrics.get('resolution_iterations', 0)
            
            self.session.add(experiment_result)
            successful_results.append(result)
        
        self.session.commit()
        
        # Calculate aggregate metrics
        if not successful_results:
            return {
                'completion_rate': 0.0, 'quality_success_rate': 0.0, 'avg_processing_time': 0.0, 'total_processing_time': 0.0,
                'avg_accuracy': 0.0, 'avg_relevancy': 0.0, 'avg_faithfulness': 0.0, 'avg_hallucination': 0.0,
                'avg_consensus_time': 0.0, 'avg_agreement_strength': 0.0
            }
        
        total_tickets = len(results)
        successful_count = len(successful_results)
        
        processing_times = [r.get('processing_time', 0.0) for r in successful_results]
        consensus_times = [r.get('collaboration_metrics', {}).get('consensus_building_duration', 0.0) for r in successful_results]
        agreement_strengths = [r.get('collaboration_metrics', {}).get('overall_agreement_strength', 0.0) for r in successful_results]
        
        # Calculate real quality metrics from actual results
        accuracy_scores = []
        relevancy_scores = []
        faithfulness_scores = []
        hallucination_scores = []
        
        for result in successful_results:
            # Extract quality metrics from actual evaluations if available
            quality_metrics = result.get('quality_metrics', {})
            if quality_metrics:
                accuracy_scores.append(quality_metrics.get('accuracy', 0))
                relevancy_scores.append(quality_metrics.get('relevancy', 0))
                faithfulness_scores.append(quality_metrics.get('faithfulness', 0))
                hallucination_scores.append(quality_metrics.get('hallucination_score', 0))
        
        # Calculate quality-based success rate (tickets with accuracy > 0.7)
        quality_success_count = sum(1 for result in successful_results 
                                  if result.get('quality_metrics', {}).get('accuracy', 0) > 0.7)
        quality_success_rate = quality_success_count / total_tickets if total_tickets > 0 else 0.0
        
        return {
            'completion_rate': float(successful_count / total_tickets),  # Renamed from success_rate
            'quality_success_rate': float(quality_success_rate),  # New quality-based metric
            'avg_processing_time': float(np.mean(processing_times)) if processing_times else 0.0,
            'total_processing_time': float(sum(processing_times)),
            'avg_accuracy': float(np.mean(accuracy_scores)) if accuracy_scores else 0.0,
            'avg_relevancy': float(np.mean(relevancy_scores)) if relevancy_scores else 0.0,
            'avg_faithfulness': float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            'avg_hallucination': float(np.mean(hallucination_scores)) if hallucination_scores else 0.0,
            'avg_consensus_time': float(np.mean(consensus_times)) if consensus_times else 0.0,
            'avg_agreement_strength': float(np.mean(agreement_strengths)) if agreement_strengths else 0.0
        }
    
    def _calculate_experiment_summary(self, experiment_id: int) -> Dict[str, int]:
        """Calculate summary statistics for an experiment."""
        runs = self.session.query(ExperimentRun).filter_by(experiment_id=experiment_id).all()
        
        total_tickets = sum(run.ticket_count for run in runs)
        successful_runs = len([run for run in runs if run.status == 'completed'])
        failed_runs = len([run for run in runs if run.status == 'failed'])
        
        return {
            'total_tickets': total_tickets,
            'successful_runs': successful_runs,
            'failed_runs': failed_runs
        }
    
    def get_experiment_results(self, experiment_id: int) -> Dict[str, Any]:
        """Get comprehensive results for an experiment."""
        experiment = self.session.query(ExperimentConfiguration).filter_by(id=experiment_id).first()
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        runs = self.session.query(ExperimentRun).filter_by(experiment_id=experiment_id).all()
        
        return {
            'experiment': experiment,
            'runs': runs,
            'summary': self._calculate_experiment_summary(experiment_id)
        }
    
    def compare_experiments(self, experiment_ids: List[int], comparison_name: str) -> Dict[str, Any]:
        """Compare multiple experiments and determine the best configuration."""
        print(f"\n📊 Comparing {len(experiment_ids)} experiments...")
        
        comparison_data = {}
        
        for exp_id in experiment_ids:
            experiment = self.session.query(ExperimentConfiguration).filter_by(id=exp_id).first()
            runs = self.session.query(ExperimentRun).filter_by(experiment_id=exp_id).all()
            
            if not experiment or not runs:
                continue
            
            # Calculate aggregate metrics across all runs
            successful_runs = [run for run in runs if run.status == 'completed']
            
            if successful_runs:
                comparison_data[exp_id] = {
                    'experiment_name': experiment.experiment_name,
                    'experiment_type': experiment.experiment_type,
                    'avg_accuracy': np.mean([run.average_accuracy for run in successful_runs if run.average_accuracy]),
                    'avg_processing_time': np.mean([run.average_processing_time for run in successful_runs if run.average_processing_time]),
                    'success_rate': np.mean([run.success_rate for run in successful_runs if run.success_rate]),
                    'avg_consensus_time': np.mean([run.average_consensus_time for run in successful_runs if run.average_consensus_time]),
                    'total_runs': len(runs),
                    'successful_runs': len(successful_runs)
                }
        
        # Determine winner based on composite score
        winner_id = None
        best_score = -1
        
        for exp_id, metrics in comparison_data.items():
            # Composite score: 40% accuracy, 30% success rate, 20% speed, 10% consensus
            accuracy_score = metrics.get('avg_accuracy', 0) or 0
            success_score = metrics.get('success_rate', 0) or 0
            speed_score = 1 / (metrics.get('avg_processing_time', 1) or 1)  # Inverse for speed
            consensus_score = 1 / (metrics.get('avg_consensus_time', 1) or 1)  # Inverse for speed
            
            # Normalize speed and consensus scores
            max_speed = max([1/(cd.get('avg_processing_time', 1) or 1) for cd in comparison_data.values()])
            max_consensus = max([1/(cd.get('avg_consensus_time', 1) or 1) for cd in comparison_data.values()])
            
            if max_speed > 0:
                speed_score = speed_score / max_speed
            if max_consensus > 0:
                consensus_score = consensus_score / max_consensus
            
            composite_score = (accuracy_score * 0.4) + (success_score * 0.3) + (speed_score * 0.2) + (consensus_score * 0.1)
            
            if composite_score > best_score:
                best_score = composite_score
                winner_id = exp_id
        
        # Save comparison to database
        comparison = ExperimentComparison(
            comparison_name=comparison_name,
            experiment_ids=experiment_ids,
            comparison_type='comprehensive',
            metric_focus='composite',
            winner_experiment_id=winner_id,
            comparison_results=comparison_data,
            recommendations=f"Best configuration: Experiment {winner_id} with composite score {best_score:.3f}"
        )
        
        self.session.add(comparison)
        self.session.commit()
        
        print(f"🏆 Winner: Experiment {winner_id} (Score: {best_score:.3f})")
        
        return {
            'comparison_id': comparison.id,
            'winner_experiment_id': winner_id,
            'comparison_data': comparison_data,
            'recommendations': comparison.recommendations
        }
    
    def close(self):
        """Close database session."""
        self.session.close()


# Convenience functions for common experiment types
def create_quick_model_comparison(test_tickets: List[Dict[str, str]]) -> ExperimentConfig:
    """Create a quick model comparison experiment."""
    manager = ExperimentManager()
    config = manager.create_model_assignment_experiment(
        name="Quick Model Comparison",
        test_tickets=test_tickets,
        agents_to_test=["ticket_analyst"],  # Test just one agent for speed
        models_to_test=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"]
    )
    manager.close()
    return config


def create_full_optimization_sweep(test_tickets: List[Dict[str, str]]) -> ExperimentConfig:
    """Create a comprehensive optimization experiment."""
    manager = ExperimentManager()
    config = manager.create_comprehensive_sweep(
        name="Full Multi-Agent Optimization",
        test_tickets=test_tickets,
        limited_scope=True  # Keep it manageable
    )
    manager.close()
    return config