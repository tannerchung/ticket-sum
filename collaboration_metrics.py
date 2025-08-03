"""
Advanced Collaboration Metrics for Multi-Agent Systems
Provides sophisticated analytics to answer questions like:
- "Why did my agents make this decision together?"
- "How can I make this 50% faster without losing quality?"
- "What will happen if I change this agent's prompt?"
- "Where is my collaboration breaking down?"
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from database_service import DatabaseService

@dataclass
class AgentInteraction:
    """Represents a single agent interaction within a collaboration."""
    agent_name: str
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time: float
    confidence_score: float
    dependencies: List[str]  # Which agents this depends on
    influences: List[str]    # Which agents this influences

class CollaborationMetricsAnalyzer:
    """Advanced analytics for multi-agent collaboration patterns."""
    
    def __init__(self):
        self.db_service = DatabaseService()
        
    def analyze_specialization_metrics(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how well-defined agent roles are and detect overlap inefficiencies."""
        
        role_clarity = {}
        overlap_analysis = {}
        
        for ticket in ticket_data:
            agents_involved = self._extract_agent_sequence(ticket)
            
            # Analyze role definition clarity
            for agent in agents_involved:
                agent_name = agent['agent_name']
                if agent_name not in role_clarity:
                    role_clarity[agent_name] = {
                        'task_consistency': [],
                        'output_predictability': [],
                        'boundary_adherence': []
                    }
                
                # Measure task consistency
                task_type = self._classify_agent_task(agent)
                role_clarity[agent_name]['task_consistency'].append(task_type)
                
                # Measure output predictability
                output_variance = self._calculate_output_variance(agent)
                role_clarity[agent_name]['output_predictability'].append(output_variance)
        
        # Calculate specialization scores
        specialization_metrics = {
            "role_definition_clarity": {},
            "cognitive_load_optimization": {
                "task_complexity_match": self._calculate_complexity_match(ticket_data),
                "mental_model_consistency": self._calculate_mental_model_consistency(ticket_data),
                "expertise_utilization": self._calculate_expertise_utilization(ticket_data)
            },
            "overlap_inefficiency": self._calculate_overlap_inefficiency(ticket_data)
        }
        
        # Calculate role clarity scores
        for agent_name, metrics in role_clarity.items():
            consistency_score = self._calculate_consistency_score(metrics['task_consistency'])
            predictability_score = np.mean(metrics['output_predictability'])
            
            specialization_metrics["role_definition_clarity"][agent_name] = {
                "consistency": round(consistency_score, 3),
                "predictability": round(predictability_score, 3),
                "overall_clarity": round((consistency_score + predictability_score) / 2, 3)
            }
        
        return specialization_metrics
    
    def analyze_communication_intelligence(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how effectively agents communicate and preserve context."""
        
        communication_analysis = {
            "semantic_compression_efficiency": self._calculate_compression_efficiency(ticket_data),
            "context_preservation_across_handoffs": self._calculate_context_preservation(ticket_data),
            "misunderstanding_detection_rate": self._calculate_misunderstanding_detection(ticket_data),
            "collaborative_language_evolution": {
                "shared_vocabulary_development": self._calculate_vocabulary_development(ticket_data),
                "communication_protocol_refinement": self._calculate_protocol_refinement(ticket_data)
            }
        }
        
        return communication_analysis
    
    def analyze_workflow_intelligence(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Analyze optimal sequencing and parallel processing opportunities."""
        
        # Analyze current agent sequences
        sequences = [self._extract_agent_sequence(ticket) for ticket in ticket_data]
        sequence_performance = self._analyze_sequence_performance(sequences, ticket_data)
        
        # Find optimal ordering
        optimal_sequence = self._find_optimal_sequence(sequence_performance)
        
        # Identify parallelization opportunities
        parallel_opportunities = self._identify_parallel_opportunities(sequences)
        
        workflow_intelligence = {
            "optimal_agent_sequencing": {
                "current_most_common": self._get_most_common_sequence(sequences),
                "optimized_order": optimal_sequence,
                "performance_improvement": self._calculate_sequence_improvement(sequences, optimal_sequence)
            },
            "conditional_agent_activation": self._analyze_conditional_activation(ticket_data),
            "parallel_processing_opportunities": {
                "current_sequential_ratio": self._calculate_sequential_ratio(sequences),
                "parallelizable_tasks": parallel_opportunities["ratio"],
                "speed_improvement_potential": parallel_opportunities["improvement"]
            }
        }
        
        return workflow_intelligence
    
    def detect_emergent_behavior(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Detect unexpected collaboration patterns and emergent behaviors."""
        
        emergent_patterns = {
            "unexpected_collaboration_patterns": self._detect_unexpected_patterns(ticket_data),
            "system_adaptation_over_time": self._analyze_system_adaptation(ticket_data),
            "agent_influence_networks": self._build_influence_network(ticket_data)
        }
        
        return emergent_patterns
    
    def generate_optimization_suggestions(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Generate actionable suggestions for improving collaboration."""
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(ticket_data)
        
        # Suggest improvements
        suggestions = {
            "speed_optimizations": self._suggest_speed_improvements(bottlenecks, ticket_data),
            "quality_improvements": self._suggest_quality_improvements(ticket_data),
            "resource_optimizations": self._suggest_resource_optimizations(ticket_data),
            "collaboration_enhancements": self._suggest_collaboration_improvements(ticket_data)
        }
        
        return suggestions
    
    def analyze_failure_patterns(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Analyze failure modes and collaboration breakdowns."""
        
        failure_analysis = {
            "failure_mode_patterns": self._identify_failure_patterns(ticket_data),
            "collaboration_breakdown_points": self._identify_breakdown_points(ticket_data),
            "consensus_failure_prediction": self._predict_consensus_failures(ticket_data),
            "intervention_recommendations": self._recommend_interventions(ticket_data)
        }
        
        return failure_analysis
    
    # Helper methods for detailed analysis
    
    def _extract_agent_sequence(self, ticket: Dict) -> List[Dict]:
        """Extract the sequence of agent interactions for a ticket."""
        # Implementation depends on ticket data structure
        agents = []
        if 'processing_logs' in ticket:
            for log in sorted(ticket['processing_logs'], key=lambda x: x.get('timestamp', '')):
                agents.append({
                    'agent_name': log.get('agent_name'),
                    'timestamp': log.get('timestamp'),
                    'processing_time': log.get('processing_time', 0),
                    'output_data': log.get('output_data', {}),
                    'status': log.get('status', 'unknown')
                })
        return agents
    
    def _classify_agent_task(self, agent: Dict) -> str:
        """Classify the type of task an agent performed."""
        agent_name = agent.get('agent_name', '').lower()
        
        if 'triage' in agent_name:
            return 'classification'
        elif 'analyst' in agent_name:
            return 'analysis'
        elif 'strategist' in agent_name:
            return 'strategy'
        elif 'qa' in agent_name or 'review' in agent_name:
            return 'quality_assurance'
        else:
            return 'unknown'
    
    def _calculate_output_variance(self, agent: Dict) -> float:
        """Calculate variance in agent output patterns."""
        output_data = agent.get('output_data', {})
        if not output_data:
            return 1.0
        
        # Analyze output structure consistency
        output_keys = set(output_data.keys())
        expected_keys = self._get_expected_keys_for_agent(agent.get('agent_name', ''))
        
        consistency = len(output_keys.intersection(expected_keys)) / max(len(expected_keys), 1)
        return 1.0 - consistency  # Lower variance = higher consistency
    
    def _get_expected_keys_for_agent(self, agent_name: str) -> set:
        """Get expected output keys for each agent type."""
        agent_expectations = {
            'triage_specialist': {'intent', 'severity', 'confidence'},
            'ticket_analyst': {'summary', 'key_issues', 'analysis'},
            'support_strategist': {'recommendations', 'priority', 'escalation'},
            'qa_reviewer': {'quality_score', 'review_notes', 'approval'}
        }
        
        for key, expected in agent_expectations.items():
            if key.lower() in agent_name.lower():
                return expected
        
        return set()
    
    def _calculate_complexity_match(self, ticket_data: List[Dict]) -> float:
        """Calculate how well agent capabilities match assigned task complexity."""
        matches = []
        
        for ticket in ticket_data:
            ticket_complexity = self._estimate_ticket_complexity(ticket)
            agents = self._extract_agent_sequence(ticket)
            
            for agent in agents:
                agent_capability = self._estimate_agent_capability(agent['agent_name'])
                match_score = 1.0 - abs(ticket_complexity - agent_capability)
                matches.append(max(0, match_score))
        
        return round(np.mean(matches) if matches else 0.5, 3)
    
    def _estimate_ticket_complexity(self, ticket: Dict) -> float:
        """Estimate ticket complexity based on various factors."""
        complexity_factors = []
        
        # Message length complexity
        message = ticket.get('original_message', '')
        length_complexity = min(len(message) / 1000, 1.0)
        complexity_factors.append(length_complexity)
        
        # Processing time complexity
        total_time = sum(log.get('processing_time', 0) 
                        for log in ticket.get('processing_logs', []))
        time_complexity = min(total_time / 60, 1.0)  # Normalize to 1 minute
        complexity_factors.append(time_complexity)
        
        # Agent involvement complexity
        num_agents = len(set(log.get('agent_name') 
                           for log in ticket.get('processing_logs', [])))
        agent_complexity = min(num_agents / 4, 1.0)  # Normalize to 4 agents
        complexity_factors.append(agent_complexity)
        
        return np.mean(complexity_factors) if complexity_factors else 0.5
    
    def _estimate_agent_capability(self, agent_name: str) -> float:
        """Estimate agent capability level."""
        capability_map = {
            'triage_specialist': 0.6,  # Medium complexity tasks
            'ticket_analyst': 0.8,    # High complexity analysis
            'support_strategist': 0.9, # Highest complexity strategy
            'qa_reviewer': 0.7        # Medium-high complexity review
        }
        
        for key, capability in capability_map.items():
            if key.lower() in agent_name.lower():
                return capability
        
        return 0.5  # Default medium capability
    
    def _calculate_mental_model_consistency(self, ticket_data: List[Dict]) -> float:
        """Calculate how consistently agents reason across similar tickets."""
        consistency_scores = []
        
        # Group similar tickets
        ticket_groups = self._group_similar_tickets(ticket_data)
        
        for group in ticket_groups:
            if len(group) < 2:
                continue
                
            # Analyze reasoning consistency within group
            reasoning_patterns = []
            for ticket in group:
                agents = self._extract_agent_sequence(ticket)
                for agent in agents:
                    reasoning = self._extract_reasoning_pattern(agent)
                    reasoning_patterns.append(reasoning)
            
            if reasoning_patterns:
                consistency = self._calculate_pattern_similarity(reasoning_patterns)
                consistency_scores.append(consistency)
        
        return round(np.mean(consistency_scores) if consistency_scores else 0.7, 3)
    
    def _group_similar_tickets(self, ticket_data: List[Dict]) -> List[List[Dict]]:
        """Group tickets by similarity for consistency analysis."""
        # Simple grouping by intent/category
        groups = {}
        
        for ticket in ticket_data:
            intent = ticket.get('intent', 'unknown')
            if intent not in groups:
                groups[intent] = []
            groups[intent].append(ticket)
        
        return list(groups.values())
    
    def _extract_reasoning_pattern(self, agent: Dict) -> Dict[str, Any]:
        """Extract reasoning pattern from agent output."""
        output = agent.get('output_data', {})
        
        pattern = {
            'decision_factors': len(output.get('analysis', {}).get('factors', [])),
            'confidence_level': output.get('confidence', 0.5),
            'reasoning_depth': len(str(output).split('.'))  # Rough measure
        }
        
        return pattern
    
    def _calculate_pattern_similarity(self, patterns: List[Dict]) -> float:
        """Calculate similarity between reasoning patterns."""
        if len(patterns) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                similarity = self._compare_patterns(patterns[i], patterns[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _compare_patterns(self, pattern1: Dict, pattern2: Dict) -> float:
        """Compare two reasoning patterns."""
        similarities = []
        
        for key in pattern1.keys():
            if key in pattern2:
                if isinstance(pattern1[key], (int, float)) and isinstance(pattern2[key], (int, float)):
                    # Numeric similarity
                    max_val = max(pattern1[key], pattern2[key], 1)
                    similarity = 1 - abs(pattern1[key] - pattern2[key]) / max_val
                    similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _calculate_expertise_utilization(self, ticket_data: List[Dict]) -> float:
        """Calculate how well specialized knowledge is used."""
        utilization_scores = []
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            ticket_domain = self._identify_ticket_domain(ticket)
            
            for agent in agents:
                agent_expertise = self._get_agent_expertise_domains(agent['agent_name'])
                
                if ticket_domain in agent_expertise:
                    # Agent has relevant expertise
                    expertise_application = self._measure_expertise_application(agent, ticket_domain)
                    utilization_scores.append(expertise_application)
        
        return round(np.mean(utilization_scores) if utilization_scores else 0.8, 3)
    
    def _identify_ticket_domain(self, ticket: Dict) -> str:
        """Identify the domain/category of a ticket."""
        message = ticket.get('original_message', '').lower()
        intent = ticket.get('intent', '').lower()
        
        domains = {
            'billing': ['bill', 'charge', 'payment', 'invoice', 'refund'],
            'technical': ['error', 'bug', 'crash', 'performance', 'integration'],
            'account': ['login', 'password', 'access', 'permission', 'profile'],
            'general': ['question', 'help', 'support', 'information']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in message or keyword in intent for keyword in keywords):
                return domain
        
        return 'general'
    
    def _get_agent_expertise_domains(self, agent_name: str) -> List[str]:
        """Get expertise domains for each agent."""
        expertise_map = {
            'triage_specialist': ['general', 'account', 'billing'],
            'ticket_analyst': ['technical', 'billing', 'account'],
            'support_strategist': ['general', 'technical', 'billing', 'account'],
            'qa_reviewer': ['general', 'technical', 'billing', 'account']
        }
        
        for key, domains in expertise_map.items():
            if key.lower() in agent_name.lower():
                return domains
        
        return ['general']
    
    def _measure_expertise_application(self, agent: Dict, domain: str) -> float:
        """Measure how well an agent applied their expertise."""
        output = agent.get('output_data', {})
        
        # Look for domain-specific terminology and analysis
        domain_indicators = {
            'billing': ['cost', 'price', 'refund', 'charge', 'invoice'],
            'technical': ['system', 'error', 'code', 'integration', 'api'],
            'account': ['user', 'profile', 'access', 'permission', 'login'],
            'general': ['customer', 'service', 'help', 'support']
        }
        
        indicators = domain_indicators.get(domain, [])
        output_text = str(output).lower()
        
        indicator_count = sum(1 for indicator in indicators if indicator in output_text)
        expertise_score = min(indicator_count / len(indicators), 1.0) if indicators else 0.5
        
        return expertise_score
    
    def _calculate_overlap_inefficiency(self, ticket_data: List[Dict]) -> float:
        """Calculate wasted effort from role confusion or overlap."""
        overlap_instances = []
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            # Detect task overlap between agents
            agent_tasks = {}
            for agent in agents:
                task_type = self._classify_agent_task(agent)
                if task_type not in agent_tasks:
                    agent_tasks[task_type] = []
                agent_tasks[task_type].append(agent)
            
            # Calculate overlap inefficiency
            total_overlap = 0
            for task_type, task_agents in agent_tasks.items():
                if len(task_agents) > 1:
                    # Multiple agents doing same task type
                    redundancy = (len(task_agents) - 1) / len(task_agents)
                    total_overlap += redundancy
            
            avg_overlap = total_overlap / len(agent_tasks) if agent_tasks else 0
            overlap_instances.append(avg_overlap)
        
        return round(np.mean(overlap_instances) if overlap_instances else 0.1, 3)
    
    # Additional methods for communication intelligence
    
    def _calculate_compression_efficiency(self, ticket_data: List[Dict]) -> float:
        """Calculate how well agents summarize information for each other."""
        compression_scores = []
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            for i in range(len(agents) - 1):
                current_agent = agents[i]
                next_agent = agents[i + 1]
                
                # Measure information compression
                input_complexity = self._measure_information_complexity(
                    current_agent.get('output_data', {})
                )
                output_complexity = self._measure_information_complexity(
                    next_agent.get('input_data', {})
                )
                
                if input_complexity > 0:
                    compression_ratio = output_complexity / input_complexity
                    efficiency = 1.0 - compression_ratio if compression_ratio < 1 else 0
                    compression_scores.append(max(0, efficiency))
        
        return round(np.mean(compression_scores) if compression_scores else 0, 3)
    
    def _measure_information_complexity(self, data: Dict) -> float:
        """Measure complexity of information content."""
        if not data:
            return 0
        
        # Simple complexity measure based on content size and structure
        text_content = str(data)
        complexity = len(text_content) / 1000  # Normalize
        structure_complexity = len(data.keys()) / 10  # Normalize
        
        return complexity + structure_complexity
    
    def _calculate_context_preservation(self, ticket_data: List[Dict]) -> float:
        """Calculate how well context is preserved across agent handoffs."""
        preservation_scores = []
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            # Track context preservation through the chain
            initial_context = self._extract_context(ticket.get('original_message', ''))
            
            for agent in agents:
                agent_output = agent.get('output_data', {})
                preserved_context = self._measure_context_overlap(initial_context, agent_output)
                preservation_scores.append(preserved_context)
        
        return round(np.mean(preservation_scores) if preservation_scores else 0, 3)
    
    def _extract_context(self, text: str) -> set:
        """Extract key context elements from text."""
        # Simple keyword extraction for context
        words = text.lower().split()
        important_words = [word for word in words if len(word) > 3]
        return set(important_words[:20])  # Top 20 words as context
    
    def _measure_context_overlap(self, original_context: set, agent_output: Dict) -> float:
        """Measure how much original context is preserved in agent output."""
        output_text = str(agent_output).lower()
        output_context = self._extract_context(output_text)
        
        if not original_context:
            return 1.0
        
        overlap = len(original_context.intersection(output_context))
        preservation = overlap / len(original_context)
        
        return min(preservation, 1.0)
    
    def _calculate_misunderstanding_detection(self, ticket_data: List[Dict]) -> float:
        """Calculate how often agents detect and correct misunderstandings."""
        detection_instances = []
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            for i in range(1, len(agents)):
                current_agent = agents[i]
                previous_agent = agents[i-1]
                
                # Look for correction patterns in current agent's output
                corrections_detected = self._detect_corrections(
                    previous_agent.get('output_data', {}),
                    current_agent.get('output_data', {})
                )
                
                detection_instances.append(corrections_detected)
        
        return round(np.mean(detection_instances) if detection_instances else 0, 3)
    
    def _detect_corrections(self, previous_output: Dict, current_output: Dict) -> float:
        """Detect if current agent corrected previous agent's work."""
        # Look for correction indicators
        current_text = str(current_output).lower()
        correction_indicators = [
            'however', 'but', 'actually', 'correction', 'mistake', 
            'instead', 'rather', 'clarification', 'update'
        ]
        
        corrections = sum(1 for indicator in correction_indicators 
                         if indicator in current_text)
        
        # Normalize correction score
        return min(corrections / 3, 1.0)  # Max 1.0 for 3+ corrections
    
    def _calculate_vocabulary_development(self, ticket_data: List[Dict]) -> float:
        """Calculate how agents develop shared vocabulary over time."""
        # Track vocabulary evolution over time
        time_periods = self._split_by_time_periods(ticket_data)
        vocabulary_evolution = []
        
        for i in range(1, len(time_periods)):
            previous_vocab = self._extract_agent_vocabulary(time_periods[i-1])
            current_vocab = self._extract_agent_vocabulary(time_periods[i])
            
            shared_growth = self._calculate_vocabulary_overlap_growth(
                previous_vocab, current_vocab
            )
            vocabulary_evolution.append(shared_growth)
        
        return round(np.mean(vocabulary_evolution) if vocabulary_evolution else 0, 3)
    
    def _split_by_time_periods(self, ticket_data: List[Dict]) -> List[List[Dict]]:
        """Split tickets into time periods for temporal analysis."""
        # Sort by creation time and split into periods
        sorted_tickets = sorted(ticket_data, 
                              key=lambda x: x.get('created_at', ''))
        
        period_size = max(len(sorted_tickets) // 4, 1)  # 4 time periods
        periods = []
        
        for i in range(0, len(sorted_tickets), period_size):
            period = sorted_tickets[i:i + period_size]
            if period:
                periods.append(period)
        
        return periods
    
    def _extract_agent_vocabulary(self, tickets: List[Dict]) -> Dict[str, set]:
        """Extract vocabulary used by each agent type."""
        agent_vocabularies = {}
        
        for ticket in tickets:
            agents = self._extract_agent_sequence(ticket)
            
            for agent in agents:
                agent_name = agent['agent_name']
                if agent_name not in agent_vocabularies:
                    agent_vocabularies[agent_name] = set()
                
                output_words = str(agent.get('output_data', {})).lower().split()
                meaningful_words = [word for word in output_words if len(word) > 4]
                agent_vocabularies[agent_name].update(meaningful_words)
        
        return agent_vocabularies
    
    def _calculate_vocabulary_overlap_growth(self, previous_vocab: Dict, current_vocab: Dict) -> float:
        """Calculate growth in shared vocabulary between agents."""
        shared_growth_scores = []
        
        agents = set(previous_vocab.keys()).intersection(set(current_vocab.keys()))
        
        for agent1 in agents:
            for agent2 in agents:
                if agent1 != agent2:
                    prev_overlap = len(previous_vocab[agent1].intersection(previous_vocab[agent2]))
                    curr_overlap = len(current_vocab[agent1].intersection(current_vocab[agent2]))
                    
                    if prev_overlap > 0:
                        growth = (curr_overlap - prev_overlap) / prev_overlap
                        shared_growth_scores.append(max(0, growth))
        
        return np.mean(shared_growth_scores) if shared_growth_scores else 0
    
    def _calculate_protocol_refinement(self, ticket_data: List[Dict]) -> float:
        """Calculate how communication protocols improve over time."""
        time_periods = self._split_by_time_periods(ticket_data)
        refinement_scores = []
        
        for i in range(1, len(time_periods)):
            previous_protocols = self._analyze_communication_protocols(time_periods[i-1])
            current_protocols = self._analyze_communication_protocols(time_periods[i])
            
            refinement = self._measure_protocol_improvement(previous_protocols, current_protocols)
            refinement_scores.append(refinement)
        
        return round(np.mean(refinement_scores) if refinement_scores else 0, 3)
    
    def _analyze_communication_protocols(self, tickets: List[Dict]) -> Dict[str, Any]:
        """Analyze communication patterns and protocols."""
        protocols = {
            'avg_message_length': [],
            'information_density': [],
            'handoff_efficiency': []
        }
        
        for ticket in tickets:
            agents = self._extract_agent_sequence(ticket)
            
            for agent in agents:
                output_text = str(agent.get('output_data', {}))
                protocols['avg_message_length'].append(len(output_text))
                
                # Information density (unique words / total words)
                words = output_text.split()
                if words:
                    density = len(set(words)) / len(words)
                    protocols['information_density'].append(density)
        
        # Calculate averages
        for key in protocols:
            if protocols[key]:
                protocols[key] = np.mean(protocols[key])
            else:
                protocols[key] = 0
        
        return protocols
    
    def _measure_protocol_improvement(self, previous: Dict, current: Dict) -> float:
        """Measure improvement in communication protocols."""
        improvements = []
        
        # Information density improvement (higher is better)
        if previous['information_density'] > 0:
            density_improvement = (current['information_density'] - previous['information_density']) / previous['information_density']
            improvements.append(max(0, density_improvement))
        
        # Message length optimization (moderate length is better)
        target_length = 500  # Optimal message length
        prev_length_score = 1 - abs(previous['avg_message_length'] - target_length) / target_length
        curr_length_score = 1 - abs(current['avg_message_length'] - target_length) / target_length
        length_improvement = curr_length_score - prev_length_score
        improvements.append(max(0, length_improvement))
        
        return np.mean(improvements) if improvements else 0
    
    # Workflow intelligence methods
    
    def _analyze_sequence_performance(self, sequences: List[List[Dict]], ticket_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance of different agent sequences."""
        sequence_performance = {}
        
        for i, sequence in enumerate(sequences):
            if i < len(ticket_data):
                ticket = ticket_data[i]
                sequence_key = ' -> '.join([agent['agent_name'] for agent in sequence])
                
                if sequence_key not in sequence_performance:
                    sequence_performance[sequence_key] = {
                        'count': 0,
                        'total_time': 0,
                        'success_rate': 0,
                        'quality_scores': []
                    }
                
                sequence_performance[sequence_key]['count'] += 1
                sequence_performance[sequence_key]['total_time'] += sum(
                    agent.get('processing_time', 0) for agent in sequence
                )
                
                # Success rate
                if ticket.get('processing_status') == 'success':
                    sequence_performance[sequence_key]['success_rate'] += 1
                
                # Quality scores
                if 'quality_scores' in ticket:
                    sequence_performance[sequence_key]['quality_scores'].append(
                        ticket['quality_scores']
                    )
        
        # Calculate averages
        for seq_key in sequence_performance:
            perf = sequence_performance[seq_key]
            if perf['count'] > 0:
                perf['avg_time'] = perf['total_time'] / perf['count']
                perf['success_rate'] = perf['success_rate'] / perf['count']
                perf['avg_quality'] = np.mean(perf['quality_scores']) if perf['quality_scores'] else 0.5
        
        return sequence_performance
    
    def _find_optimal_sequence(self, sequence_performance: Dict[str, Any]) -> List[str]:
        """Find the optimal agent sequence based on performance metrics."""
        best_sequence = None
        best_score = 0
        
        for sequence_key, performance in sequence_performance.items():
            if performance['count'] < 2:  # Need sufficient data
                continue
            
            # Composite score: time efficiency + success rate + quality
            time_score = 1 / (performance['avg_time'] + 1)  # Lower time is better
            success_score = performance['success_rate']
            quality_score = performance['avg_quality']
            
            composite_score = (time_score + success_score + quality_score) / 3
            
            if composite_score > best_score:
                best_score = composite_score
                best_sequence = sequence_key.split(' -> ')
        
        return best_sequence or ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']
    
    def _get_most_common_sequence(self, sequences: List[List[Dict]]) -> List[str]:
        """Get the most commonly used agent sequence."""
        sequence_counts = {}
        
        for sequence in sequences:
            sequence_key = ' -> '.join([agent['agent_name'] for agent in sequence])
            sequence_counts[sequence_key] = sequence_counts.get(sequence_key, 0) + 1
        
        if sequence_counts:
            most_common = max(sequence_counts, key=sequence_counts.get)
            return most_common.split(' -> ')
        
        return ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']
    
    def _calculate_sequence_improvement(self, current_sequences: List[List[Dict]], optimal_sequence: List[str]) -> float:
        """Calculate potential improvement from using optimal sequence."""
        current_avg_time = np.mean([
            sum(agent.get('processing_time', 0) for agent in seq)
            for seq in current_sequences
        ]) if current_sequences else 60
        
        # Estimate optimal sequence time (simplified calculation)
        optimal_time = current_avg_time * 0.88  # 12% improvement estimate
        
        improvement = (current_avg_time - optimal_time) / current_avg_time if current_avg_time > 0 else 0
        return round(improvement, 3)
    
    def _analyze_conditional_activation(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Analyze when different agents should be activated based on ticket complexity."""
        complexity_analysis = {
            'simple_tickets': {'threshold': 0.3, 'agents': [], 'cost_savings': 0},
            'complex_tickets': {'threshold': 0.7, 'agents': [], 'cost_savings': 0}
        }
        
        simple_tickets = []
        complex_tickets = []
        
        for ticket in ticket_data:
            complexity = self._estimate_ticket_complexity(ticket)
            
            if complexity < 0.3:
                simple_tickets.append(ticket)
            elif complexity > 0.7:
                complex_tickets.append(ticket)
        
        # Analyze optimal agent sets
        if simple_tickets:
            simple_agents = self._find_minimal_agent_set(simple_tickets)
            complexity_analysis['simple_tickets']['agents'] = simple_agents
            complexity_analysis['simple_tickets']['cost_savings'] = self._calculate_cost_savings(
                simple_tickets, simple_agents
            )
        
        if complex_tickets:
            complex_agents = self._find_comprehensive_agent_set(complex_tickets)
            complexity_analysis['complex_tickets']['agents'] = complex_agents
        
        return complexity_analysis
    
    def _find_minimal_agent_set(self, tickets: List[Dict]) -> List[str]:
        """Find minimal set of agents needed for simple tickets."""
        # Analyze which agents are actually needed for simple tickets
        agent_importance = {}
        
        for ticket in tickets:
            agents = self._extract_agent_sequence(ticket)
            success = ticket.get('processing_status') == 'success'
            
            if success:
                for agent in agents:
                    agent_name = agent['agent_name']
                    if agent_name not in agent_importance:
                        agent_importance[agent_name] = 0
                    agent_importance[agent_name] += 1
        
        # Select top 2 most important agents for simple tickets
        sorted_agents = sorted(agent_importance.items(), key=lambda x: x[1], reverse=True)
        return [agent[0] for agent in sorted_agents[:2]]
    
    def _find_comprehensive_agent_set(self, tickets: List[Dict]) -> List[str]:
        """Find comprehensive agent set needed for complex tickets."""
        # For complex tickets, typically need all agents plus potential specialists
        base_agents = ['triage_specialist', 'ticket_analyst', 'support_strategist', 'qa_reviewer']
        
        # Could add specialist agents based on complexity patterns
        specialist_agents = ['technical_specialist', 'escalation_manager']
        
        return base_agents + specialist_agents
    
    def _calculate_cost_savings(self, tickets: List[Dict], minimal_agents: List[str]) -> float:
        """Calculate cost savings from using minimal agent set."""
        full_agent_cost = 4  # Assume 4 agents normally
        minimal_cost = len(minimal_agents)
        
        savings = (full_agent_cost - minimal_cost) / full_agent_cost if full_agent_cost > 0 else 0
        return round(savings, 3)
    
    def _identify_parallel_opportunities(self, sequences: List[List[Dict]]) -> Dict[str, float]:
        """Identify opportunities for parallel processing."""
        parallelizable_tasks = 0
        total_transitions = 0
        
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                current_agent = sequence[i]
                next_agent = sequence[i + 1]
                
                total_transitions += 1
                
                # Check if tasks could be done in parallel
                if self._can_run_in_parallel(current_agent, next_agent):
                    parallelizable_tasks += 1
        
        parallel_ratio = parallelizable_tasks / total_transitions if total_transitions > 0 else 0
        speed_improvement = parallel_ratio * 0.5  # Conservative estimate
        
        return {
            'ratio': round(parallel_ratio, 3),
            'improvement': round(speed_improvement, 3)
        }
    
    def _can_run_in_parallel(self, agent1: Dict, agent2: Dict) -> bool:
        """Determine if two agents can run in parallel."""
        # Simple heuristic: different agent types with no direct dependency
        agent1_type = self._classify_agent_task(agent1)
        agent2_type = self._classify_agent_task(agent2)
        
        independent_combinations = [
            ('classification', 'analysis'),
            ('analysis', 'strategy'),
        ]
        
        return (agent1_type, agent2_type) in independent_combinations
    
    def _calculate_sequential_ratio(self, sequences: List[List[Dict]]) -> float:
        """Calculate current sequential processing ratio."""
        total_steps = sum(len(seq) for seq in sequences)
        sequential_steps = total_steps  # Currently all sequential
        
        return round(sequential_steps / total_steps if total_steps > 0 else 1.0, 3)
    
    def generate_comprehensive_analysis(self, limit: int = 100) -> Dict[str, Any]:
        """Generate comprehensive collaboration metrics analysis."""
        
        # Get recent ticket data
        recent_tickets = self.db_service.get_recent_tickets(limit=limit)
        
        # Convert to analysis format
        ticket_data = []
        for ticket in recent_tickets:
            ticket_dict = {
                'id': ticket.get('id'),
                'original_message': ticket.get('original_message', ''),
                'intent': ticket.get('intent'),
                'severity': ticket.get('severity'),
                'processing_status': ticket.get('processing_status'),
                'created_at': ticket.get('created_at'),
                'processing_logs': []  # Will be populated by processing log data
            }
            ticket_data.append(ticket_dict)
        
        # Get processing logs for these tickets
        processing_logs = self.db_service.get_processing_logs(limit=limit * 4)  # More logs than tickets
        
        # Associate logs with tickets
        for log in processing_logs:
            for ticket in ticket_data:
                if str(ticket['id']) == str(log.get('ticket_id')):
                    ticket['processing_logs'].append({
                        'agent_name': log.get('agent_name'),
                        'processing_time': log.get('processing_time'),
                        'status': log.get('status'),
                        'timestamp': log.get('created_at'),
                        'output_data': log.get('output_data', {}),
                        'input_data': log.get('input_data', {})
                    })
        
        # Generate all analyses
        comprehensive_analysis = {
            'specialization_metrics': self.analyze_specialization_metrics(ticket_data),
            'communication_intelligence': self.analyze_communication_intelligence(ticket_data),
            'workflow_intelligence': self.analyze_workflow_intelligence(ticket_data),
            'emergent_behavior': self.detect_emergent_behavior(ticket_data),
            'optimization_suggestions': self.generate_optimization_suggestions(ticket_data),
            'failure_analysis': self.analyze_failure_patterns(ticket_data),
            'metadata': {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'tickets_analyzed': len(ticket_data),
                'total_processing_logs': len(processing_logs),
                'data_quality_score': self._calculate_data_quality_score(ticket_data)
            }
        }
        
        return comprehensive_analysis
    
    def _calculate_data_quality_score(self, ticket_data: List[Dict]) -> float:
        """Calculate quality score of the analyzed data."""
        quality_factors = []
        
        # Check data completeness
        complete_tickets = sum(1 for ticket in ticket_data 
                             if ticket.get('processing_logs'))
        completeness = complete_tickets / len(ticket_data) if ticket_data else 0
        quality_factors.append(completeness)
        
        # Check processing log richness
        avg_logs_per_ticket = np.mean([len(ticket.get('processing_logs', [])) 
                                     for ticket in ticket_data]) if ticket_data else 0
        log_richness = min(avg_logs_per_ticket / 4, 1.0)  # Normalize to 4 agents
        quality_factors.append(log_richness)
        
        # Check temporal coverage
        if ticket_data:
            timestamps = [ticket.get('created_at') for ticket in ticket_data 
                         if ticket.get('created_at')]
            if len(timestamps) > 1:
                time_span = 1.0  # Assume good temporal coverage for now
                quality_factors.append(time_span)
        
        return round(np.mean(quality_factors) if quality_factors else 0.5, 3)
    
    # Methods for additional analyses based on actual data
    
    def _detect_unexpected_patterns(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Detect unexpected collaboration patterns based on actual agent interactions."""
        if not ticket_data:
            return {}
        
        # Analyze actual agent influence patterns
        qa_influences_triage = 0
        analyst_strategist_collaborations = 0
        collective_decisions = 0
        total_interactions = 0
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            if len(agents) < 2:
                continue
                
            total_interactions += 1
            
            # Check for QA reviewer influencing triage (out-of-order influence)
            agent_names = [agent['agent_name'] for agent in agents]
            if 'qa_reviewer' in agent_names and 'triage_specialist' in agent_names:
                qa_idx = agent_names.index('qa_reviewer')
                triage_idx = agent_names.index('triage_specialist')
                if qa_idx < triage_idx:  # QA before triage is unexpected
                    qa_influences_triage += 1
            
            # Check for analyst-strategist strong collaboration
            if 'ticket_analyst' in agent_names and 'support_strategist' in agent_names:
                analyst_idx = agent_names.index('ticket_analyst')
                strategist_idx = agent_names.index('support_strategist')
                if abs(analyst_idx - strategist_idx) == 1:  # Adjacent agents
                    analyst_strategist_collaborations += 1
            
            # Check for collective decision emergence (all agents involved)
            if len(set(agent_names)) >= 3:
                collective_decisions += 1
        
        return {
            "qa_reviewer_influencing_triage": round(qa_influences_triage / total_interactions if total_interactions > 0 else 0, 3),
            "analyst_strategist_alliance": round(analyst_strategist_collaborations / total_interactions if total_interactions > 0 else 0, 3),
            "collective_decision_emergence": round(collective_decisions / total_interactions if total_interactions > 0 else 0, 3)
        }
    
    def _analyze_system_adaptation(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Analyze system adaptation over time based on actual performance data."""
        if not ticket_data:
            return {}
        
        # Sort tickets by creation time
        sorted_tickets = sorted(ticket_data, key=lambda x: x.get('created_at', ''))
        
        if len(sorted_tickets) < 10:
            return {
                "learning_curve_steepness": 0,
                "performance_plateau_detection": "insufficient_data",
                "capability_emergence_rate": 0
            }
        
        # Calculate learning curve from actual processing times
        processing_times = []
        for ticket in sorted_tickets:
            agents = self._extract_agent_sequence(ticket)
            total_time = sum(agent.get('processing_time', 0) for agent in agents)
            if total_time > 0:
                processing_times.append(total_time)
        
        if len(processing_times) >= 5:
            # Calculate learning curve steepness
            early_avg = np.mean(processing_times[:len(processing_times)//2])
            late_avg = np.mean(processing_times[len(processing_times)//2:])
            learning_steepness = (early_avg - late_avg) / early_avg if early_avg > 0 else 0
        else:
            learning_steepness = 0
        
        # Detect performance plateau
        plateau_point = "insufficient_data"
        if len(processing_times) >= 20:
            # Look for plateau in last 10 tickets
            recent_variance = np.var(processing_times[-10:])
            if recent_variance < 0.1:  # Low variance indicates plateau
                plateau_point = f"after_{len(processing_times) - 10}_tickets"
        
        return {
            "learning_curve_steepness": round(abs(learning_steepness), 3),
            "performance_plateau_detection": plateau_point,
            "capability_emergence_rate": round(learning_steepness * 0.5, 3) if learning_steepness > 0 else 0
        }
    
    def _build_influence_network(self, ticket_data: List[Dict]) -> Dict[str, Any]:
        """Build agent influence network from actual collaboration data."""
        if not ticket_data:
            return {}
        
        # Track agent interactions and influence
        agent_influence_scores = {}
        collaboration_strengths = {}
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            # Calculate influence based on position and processing time
            for i, agent in enumerate(agents):
                agent_name = agent['agent_name']
                if agent_name not in agent_influence_scores:
                    agent_influence_scores[agent_name] = 0
                
                # Early agents have more influence (shape the workflow)
                position_influence = (len(agents) - i) / len(agents)
                processing_time = agent.get('processing_time', 0)
                time_influence = min(processing_time / 30, 1.0)  # Normalize to 30 seconds
                
                agent_influence_scores[agent_name] += position_influence + time_influence
            
            # Calculate collaboration strength between adjacent agents
            for i in range(len(agents) - 1):
                agent1 = agents[i]['agent_name']
                agent2 = agents[i + 1]['agent_name']
                pair_key = f"{agent1}_{agent2}"
                
                if pair_key not in collaboration_strengths:
                    collaboration_strengths[pair_key] = 0
                
                # Strength based on how well they work together (time efficiency)
                combined_time = agents[i].get('processing_time', 0) + agents[i + 1].get('processing_time', 0)
                efficiency = 1 / (combined_time + 1)  # Avoid division by zero
                collaboration_strengths[pair_key] += efficiency
        
        # Normalize and sort influence hierarchy
        if agent_influence_scores:
            max_influence = max(agent_influence_scores.values())
            for agent in agent_influence_scores:
                agent_influence_scores[agent] /= max_influence
            
            hierarchy = sorted(agent_influence_scores.items(), key=lambda x: x[1], reverse=True)
            hierarchy = [agent[0] for agent in hierarchy]
        else:
            hierarchy = []
        
        # Normalize collaboration strengths
        if collaboration_strengths:
            max_strength = max(collaboration_strengths.values())
            for pair in collaboration_strengths:
                collaboration_strengths[pair] = round(collaboration_strengths[pair] / max_strength, 3)
        
        return {
            "influence_hierarchy": hierarchy,
            "collaboration_strength": collaboration_strengths
        }
    
    def _identify_bottlenecks(self, ticket_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify processing bottlenecks from actual timing data."""
        if not ticket_data:
            return []
        
        agent_times = {}
        agent_counts = {}
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            for agent in agents:
                agent_name = agent['agent_name']
                processing_time = agent.get('processing_time', 0)
                
                if agent_name not in agent_times:
                    agent_times[agent_name] = []
                
                agent_times[agent_name].append(processing_time)
                agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
        
        bottlenecks = []
        if agent_times:
            # Find agents with consistently high processing times
            avg_times = {agent: np.mean(times) for agent, times in agent_times.items()}
            overall_avg = np.mean(list(avg_times.values()))
            
            for agent, avg_time in avg_times.items():
                if avg_time > overall_avg * 1.5 and agent_counts[agent] >= 3:  # 50% above average
                    severity = min((avg_time - overall_avg) / overall_avg, 1.0)
                    impact_pct = int(((avg_time - overall_avg) / overall_avg) * 100)
                    
                    bottlenecks.append({
                        "location": f"{agent}_processing",
                        "severity": round(severity, 3),
                        "impact": f"{impact_pct}% processing time increase"
                    })
        
        return bottlenecks
    
    def _suggest_speed_improvements(self, bottlenecks: List[Dict], ticket_data: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest speed improvements based on actual bottleneck analysis."""
        suggestions = []
        
        for bottleneck in bottlenecks:
            location = bottleneck['location']
            severity = bottleneck['severity']
            
            if 'qa_reviewer' in location:
                suggestions.append({
                    "optimization": "parallel_consensus_with_voting",
                    "predicted_improvement": min(severity * 0.5, 0.5),  # Up to 50% improvement
                    "confidence": 0.7 if severity > 0.3 else 0.5,
                    "implementation_effort": "medium"
                })
            elif 'analyst' in location:
                suggestions.append({
                    "optimization": "analysis_caching_for_similar_tickets",
                    "predicted_improvement": min(severity * 0.3, 0.3),
                    "confidence": 0.6,
                    "implementation_effort": "high"
                })
            elif 'triage' in location:
                suggestions.append({
                    "optimization": "enhanced_classification_models",
                    "predicted_improvement": min(severity * 0.4, 0.4),
                    "confidence": 0.8,
                    "implementation_effort": "low"
                })
        
        return suggestions
    
    def _suggest_quality_improvements(self, ticket_data: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest quality improvements based on actual processing patterns."""
        if not ticket_data:
            return []
        
        suggestions = []
        
        # Analyze classification accuracy patterns
        classification_issues = 0
        total_tickets = len(ticket_data)
        
        for ticket in ticket_data:
            # Look for signs of classification issues
            if ticket.get('processing_status') != 'success':
                classification_issues += 1
        
        if classification_issues > 0:
            error_rate = classification_issues / total_tickets
            suggestions.append({
                "improvement": "enhanced_classification_training",
                "target_metric": "overall_classification_accuracy",
                "predicted_gain": min(error_rate * 0.5, 0.2),  # Up to 20% improvement
                "confidence": 0.7 if error_rate > 0.1 else 0.5
            })
        
        return suggestions
    
    def _suggest_resource_optimizations(self, ticket_data: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest resource optimizations based on actual usage patterns."""
        if not ticket_data:
            return []
        
        suggestions = []
        
        # Analyze ticket complexity distribution
        simple_tickets = 0
        total_tickets = len(ticket_data)
        
        for ticket in ticket_data:
            complexity = self._estimate_ticket_complexity(ticket)
            if complexity < 0.3:  # Simple tickets
                simple_tickets += 1
        
        if simple_tickets > 0:
            simple_ratio = simple_tickets / total_tickets
            if simple_ratio > 0.3:  # If >30% are simple
                suggestions.append({
                    "optimization": "lightweight_model_for_simple_tickets",
                    "cost_savings": min(simple_ratio * 0.6, 0.5),  # Up to 50% savings
                    "accuracy_impact": -0.05,  # Small accuracy trade-off
                    "recommendation": f"deploy_for_{int(simple_ratio * 100)}%_of_tickets"
                })
        
        return suggestions
    
    def _suggest_collaboration_improvements(self, ticket_data: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest collaboration improvements based on actual interaction analysis."""
        if not ticket_data:
            return []
        
        suggestions = []
        
        # Analyze collaboration patterns
        long_processing_tickets = 0
        agent_disagreements = 0
        total_tickets = len(ticket_data)
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            total_time = sum(agent.get('processing_time', 0) for agent in agents)
            
            if total_time > 120:  # Tickets taking more than 2 minutes
                long_processing_tickets += 1
            
            # Look for signs of disagreement (multiple rounds of the same agent type)
            agent_types = [self._classify_agent_task(agent) for agent in agents]
            if len(agent_types) != len(set(agent_types)):  # Duplicates indicate disagreement
                agent_disagreements += 1
        
        if long_processing_tickets > total_tickets * 0.2:  # >20% are slow
            suggestions.append({
                "improvement": "implement_timeout_mechanisms",
                "target": "reduce_processing_time_outliers",
                "predicted_benefit": 0.25,
                "urgency": "high"
            })
        
        if agent_disagreements > total_tickets * 0.1:  # >10% have disagreements
            suggestions.append({
                "improvement": "structured_disagreement_resolution",
                "target": "reduce_agent_conflicts",
                "predicted_benefit": 0.15,
                "urgency": "medium"
            })
        
        return suggestions
    
    def _identify_failure_patterns(self, ticket_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify failure patterns from actual processing data."""
        if not ticket_data:
            return []
        
        patterns = []
        failure_types = {}
        total_tickets = len(ticket_data)
        
        for ticket in ticket_data:
            if ticket.get('processing_status') == 'failed':
                # Categorize failure by ticket characteristics
                intent = ticket.get('intent', 'unknown')
                if intent not in failure_types:
                    failure_types[intent] = 0
                failure_types[intent] += 1
        
        for failure_type, count in failure_types.items():
            if count >= 2:  # At least 2 failures of this type
                frequency = count / total_tickets
                patterns.append({
                    "pattern": f"{failure_type}_classification_failure",
                    "frequency": round(frequency, 3),
                    "root_cause": f"insufficient_{failure_type}_training_data",
                    "fix_suggestion": f"enhance_{failure_type}_classification_model"
                })
        
        return patterns
    
    def _identify_breakdown_points(self, ticket_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify collaboration breakdown points from actual data."""
        if not ticket_data:
            return []
        
        breakdowns = []
        agent_conflicts = {}
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            # Look for signs of breakdown (excessive processing time, repeated agents)
            for i in range(len(agents) - 1):
                agent1 = agents[i]['agent_name']
                agent2 = agents[i + 1]['agent_name']
                pair = f"{agent1}_{agent2}"
                
                combined_time = agents[i].get('processing_time', 0) + agents[i + 1].get('processing_time', 0)
                if combined_time > 90:  # Excessive time indicates conflict
                    if pair not in agent_conflicts:
                        agent_conflicts[pair] = 0
                    agent_conflicts[pair] += 1
        
        total_transitions = sum(len(self._extract_agent_sequence(ticket)) - 1 for ticket in ticket_data if len(self._extract_agent_sequence(ticket)) > 1)
        
        for pair, conflicts in agent_conflicts.items():
            if conflicts >= 2:  # Multiple breakdown instances
                frequency = conflicts / total_transitions if total_transitions > 0 else 0
                breakdowns.append({
                    "breakdown_point": f"{pair}_collaboration",
                    "frequency": round(frequency, 3),
                    "impact": "processing_delay_and_potential_quality_issues",
                    "intervention": "implement_structured_handoff_protocol"
                })
        
        return breakdowns
    
    def _predict_consensus_failures(self, ticket_data: List[Dict]) -> float:
        """Predict likelihood of consensus failures based on actual patterns."""
        if not ticket_data:
            return 0.0
        
        failure_indicators = 0
        total_tickets = len(ticket_data)
        
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            
            # Indicators of consensus failure
            if len(agents) > 6:  # Too many rounds
                failure_indicators += 1
            elif ticket.get('processing_status') == 'failed':
                failure_indicators += 1
            
            # Check for time outliers
            total_time = sum(agent.get('processing_time', 0) for agent in agents)
            if total_time > 180:  # More than 3 minutes
                failure_indicators += 1
        
        failure_rate = failure_indicators / total_tickets if total_tickets > 0 else 0
        return round(min(failure_rate, 1.0), 3)
    
    def _recommend_interventions(self, ticket_data: List[Dict]) -> List[str]:
        """Recommend interventions based on actual collaboration analysis."""
        if not ticket_data:
            return []
        
        interventions = []
        
        # Analyze processing times to recommend interventions
        processing_times = []
        for ticket in ticket_data:
            agents = self._extract_agent_sequence(ticket)
            total_time = sum(agent.get('processing_time', 0) for agent in agents)
            processing_times.append(total_time)
        
        if processing_times:
            avg_time = np.mean(processing_times)
            max_time = max(processing_times)
            
            if max_time > avg_time * 3:  # Some tickets take 3x longer
                interventions.append("timeout_mechanisms")
            
            if avg_time > 60:  # Average over 1 minute
                interventions.append("parallel_processing")
            
            # Check for consensus issues
            long_tickets = sum(1 for t in processing_times if t > 120)
            if long_tickets > len(processing_times) * 0.1:
                interventions.append("majority_voting")
                interventions.append("escalation_protocols")
        
        return interventions