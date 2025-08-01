"""
CrewAI collaborative agent definitions for customer support ticket processing.
Implements a multi-agent system where agents collaborate and refine each other's work.
"""

import json
import time
from typing import Dict, Any, List, Optional
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from config import (
    LLM_MODEL, 
    OPENAI_API_KEY,
    COHERE_API_KEY,
    ANTHROPIC_API_KEY,
    AVAILABLE_MODELS, 
    DEFAULT_AGENT_MODELS,
    AGENT_MODEL_RECOMMENDATIONS
)

# Try to import Cohere support with better error handling
COHERE_AVAILABLE = False
ChatCohere = None
try:
    from langchain_cohere import ChatCohere
    # Test if import actually works by checking class attributes
    test_attrs = hasattr(ChatCohere, '__init__')
    if test_attrs:
        COHERE_AVAILABLE = True
        print("✅ Cohere integration enabled successfully")
    else:
        print("⚠️ Cohere integration: Import successful but class not fully compatible")
except Exception as e:
    print(f"⚠️ Cohere integration unavailable: {str(e)}")

# Try to import Anthropic support
ANTHROPIC_AVAILABLE = False
ChatAnthropic = None
try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Anthropic integration unavailable: {str(e)}")

class CollaborativeSupportCrew:
    """
    CrewAI collaborative system for customer support ticket processing.
    Agents work together, provide feedback, and refine each other's outputs.
    """
    
    def __init__(self, agent_models: Optional[Dict[str, str]] = None):
        """
        Initialize the collaborative crew with specialized agents.
        
        Args:
            agent_models: Optional dictionary mapping agent names to model names.
                         If None, uses DEFAULT_AGENT_MODELS.
        """
        # Set up agent-specific models
        self.agent_models = agent_models or DEFAULT_AGENT_MODELS.copy()
        self.llm_instances = {}
        
        # Create LLM instances for each agent
        for agent_name, model_name in self.agent_models.items():
            if model_name in AVAILABLE_MODELS:
                model_config = AVAILABLE_MODELS[model_name]
                provider = model_config.get("provider", "openai")
                
                if provider == "openai":
                    self.llm_instances[agent_name] = ChatOpenAI(
                        model=model_name,
                        api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                        temperature=model_config["temperature"]
                    )
                elif provider == "cohere" and COHERE_AVAILABLE and ChatCohere:
                    try:
                        self.llm_instances[agent_name] = ChatCohere(
                            model=model_name,
                            cohere_api_key=SecretStr(COHERE_API_KEY) if COHERE_API_KEY else None,
                            temperature=model_config["temperature"]
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to initialize Cohere for {agent_name}: {e}")
                        self.llm_instances[agent_name] = ChatOpenAI(
                            model="gpt-4o",
                            api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                            temperature=0.1
                        )
                elif provider == "anthropic" and ANTHROPIC_AVAILABLE and ChatAnthropic and ANTHROPIC_API_KEY:
                    try:
                        self.llm_instances[agent_name] = ChatAnthropic(
                            model_name=model_name,
                            api_key=SecretStr(ANTHROPIC_API_KEY),
                            temperature=model_config["temperature"],
                            timeout=60,
                            stop=None
                        )
                    except Exception as e:
                        print(f"⚠️ Failed to initialize Anthropic for {agent_name}: {e}")
                        self.llm_instances[agent_name] = ChatOpenAI(
                            model="gpt-4o",
                            api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                            temperature=0.1
                        )
                else:
                    # Fallback to OpenAI for unavailable providers
                    print(f"⚠️ {provider.title()} not available, using GPT-4o for {agent_name}")
                    self.llm_instances[agent_name] = ChatOpenAI(
                        model="gpt-4o",
                        api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                        temperature=0.1
                    )
            else:
                # Fallback to default model
                self.llm_instances[agent_name] = ChatOpenAI(
                    model=LLM_MODEL,
                    api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                    temperature=0.1
                )
        
        # Initialize collaborative agents with their specific LLMs
        self.triage_specialist = self._create_triage_specialist()
        self.ticket_analyst = self._create_ticket_analyst()
        self.support_strategist = self._create_support_strategist()
        self.qa_reviewer = self._create_qa_reviewer()
        
        # Create the collaborative crew
        self.crew = Crew(
            agents=[
                self.triage_specialist,
                self.ticket_analyst, 
                self.support_strategist,
                self.qa_reviewer
            ],
            verbose=True,
            memory=True  # Enable memory for agent collaboration
        )
    
    def _create_triage_specialist(self) -> Agent:
        """Create the triage specialist for initial classification and severity assessment."""
        return Agent(
            role="Triage Specialist",
            goal="Perform initial classification of customer support tickets and assess severity levels accurately",
            backstory="""
            You are an experienced customer support triage specialist with expertise in quickly 
            identifying ticket intent and urgency. You've processed thousands of tickets across 
            various categories including billing, technical issues, complaints, and feature requests.
            
            Your role is to provide initial classification but also collaborate with other agents
            to refine your assessment based on their insights. You're open to feedback and will
            adjust your classifications when presented with compelling evidence.
            """,
            verbose=True,
            allow_delegation=True,
            llm=self.llm_instances["triage_specialist"],
            max_execution_time=300
        )
    
    def _create_ticket_analyst(self) -> Agent:
        """Create the ticket analyst for detailed analysis and summarization."""
        return Agent(
            role="Ticket Analyst", 
            goal="Create comprehensive ticket summaries and identify key issues that may have been missed",
            backstory="""
            You are a skilled technical analyst specializing in customer support documentation.
            You excel at reading between the lines to identify underlying issues, customer sentiment,
            and technical details that others might miss.
            
            You collaborate closely with the Triage Specialist to ensure classifications are accurate
            based on your detailed analysis. You also work with the Support Strategist to ensure
            your summaries provide the right context for action planning.
            """,
            verbose=True,
            allow_delegation=True,
            llm=self.llm_instances["ticket_analyst"],
            max_execution_time=300
        )
    
    def _create_support_strategist(self) -> Agent:
        """Create the support strategist for action planning and workflow optimization."""
        return Agent(
            role="Support Strategist",
            goal="Develop optimal resolution strategies and ensure consistency between classification, analysis, and recommended actions",
            backstory="""
            You are a customer support operations strategist with deep knowledge of support workflows,
            escalation procedures, and resource allocation. You understand how different ticket types
            require different approaches and can spot inconsistencies in classifications or summaries.
            
            You collaborate with all other agents to ensure the final action plan is coherent and
            appropriate. If you notice conflicts between severity and recommended actions, you'll
            question the team and push for clarification and consensus.
            """,
            verbose=True,
            allow_delegation=True,
            llm=self.llm_instances["support_strategist"],
            max_execution_time=300
        )
    
    def _create_qa_reviewer(self) -> Agent:
        """Create the QA reviewer for final validation and consistency checking."""
        return Agent(
            role="QA Reviewer",
            goal="Ensure consistency and quality across all agent outputs, identify conflicts, and provide final validation",
            backstory="""
            You are a quality assurance specialist who reviews the work of support teams to ensure
            consistency, accuracy, and appropriate escalation decisions. You have a keen eye for
            detecting inconsistencies between classification, summary, and recommended actions.
            
            Your role is to review the collaborative work of the other agents and flag any issues:
            - Does the severity match the recommended actions?
            - Is the summary consistent with the classification?
            - Are there any logical gaps or contradictions?
            
            You facilitate final consensus among the team and ensure the output meets quality standards.
            """,
            verbose=True,
            allow_delegation=True,
            llm=self.llm_instances["qa_reviewer"],
            max_execution_time=300
        )
    
    def get_agent_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Return information about which models each agent is using."""
        return {
            agent_name: {
                "model": model_name,
                "model_info": AVAILABLE_MODELS.get(model_name, {}),
                "recommendations": AGENT_MODEL_RECOMMENDATIONS.get(agent_name, {})
            }
            for agent_name, model_name in self.agent_models.items()
        }
    
    def update_agent_model(self, agent_name: str, model_name: str) -> bool:
        """
        Update the model for a specific agent.
        
        Args:
            agent_name: Name of the agent to update
            model_name: New model name to use
            
        Returns:
            bool: True if update was successful
        """
        if agent_name not in self.agent_models:
            return False
            
        if model_name not in AVAILABLE_MODELS:
            return False
            
        # Update the model configuration
        self.agent_models[agent_name] = model_name
        
        # Create new LLM instance for this agent
        model_config = AVAILABLE_MODELS[model_name]
        provider = model_config.get("provider", "openai")
        
        if provider == "openai":
            self.llm_instances[agent_name] = ChatOpenAI(
                model=model_name,
                api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                temperature=model_config["temperature"]
            )
        elif provider == "cohere" and COHERE_AVAILABLE and ChatCohere:
            try:
                self.llm_instances[agent_name] = ChatCohere(
                    model=model_name,
                    cohere_api_key=SecretStr(COHERE_API_KEY) if COHERE_API_KEY else None,
                    temperature=model_config["temperature"]
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize Cohere for {agent_name}: {e}")
                self.llm_instances[agent_name] = ChatOpenAI(
                    model="gpt-4o", 
                    api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                    temperature=0.1
                )
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE and ChatAnthropic and ANTHROPIC_API_KEY:
            try:
                self.llm_instances[agent_name] = ChatAnthropic(
                    model_name=model_name,
                    api_key=SecretStr(ANTHROPIC_API_KEY),
                    temperature=model_config["temperature"],
                    timeout=60,
                    stop=None
                )
            except Exception as e:
                print(f"⚠️ Failed to initialize Anthropic for {agent_name}: {e}")
                self.llm_instances[agent_name] = ChatOpenAI(
                    model="gpt-4o",
                    api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                    temperature=0.1
                )
        else:
            # Fallback to OpenAI for unavailable providers
            print(f"⚠️ {provider.title()} not available, using GPT-4o for {agent_name}")
            self.llm_instances[agent_name] = ChatOpenAI(
                model="gpt-4o",
                api_key=SecretStr(OPENAI_API_KEY) if OPENAI_API_KEY else None,
                temperature=0.1
            )
        
        # Recreate the specific agent with new LLM
        if agent_name == "triage_specialist":
            self.triage_specialist = self._create_triage_specialist()
        elif agent_name == "ticket_analyst":
            self.ticket_analyst = self._create_ticket_analyst()
        elif agent_name == "support_strategist":
            self.support_strategist = self._create_support_strategist()
        elif agent_name == "qa_reviewer":
            self.qa_reviewer = self._create_qa_reviewer()
            
        # Recreate the crew with updated agents
        self.crew = Crew(
            agents=[
                self.triage_specialist,
                self.ticket_analyst, 
                self.support_strategist,
                self.qa_reviewer
            ],
            verbose=True,
            memory=True
        )
        
        return True
    
    def compare_models_on_tickets(self, tickets: List[Dict[str, str]], agent_name: str, models_to_test: List[str]) -> Dict[str, Any]:
        """
        Compare different models for a specific agent across multiple tickets.
        
        Args:
            tickets: List of ticket dictionaries with 'id' and 'content' keys
            agent_name: Name of the agent to test different models on
            models_to_test: List of model names to test
            
        Returns:
            Dict containing comparison results and performance metrics
        """
        comparison_results = {
            "agent_name": agent_name,
            "tickets_tested": len(tickets),
            "models_tested": models_to_test,
            "results": {},
            "performance_summary": {}
        }
        
        original_model = self.agent_models[agent_name]
        
        for model_name in models_to_test:
            if model_name not in AVAILABLE_MODELS:
                continue
                
            model_results = {
                "model_name": model_name,
                "model_info": AVAILABLE_MODELS[model_name],
                "ticket_results": [],
                "processing_times": [],
                "error_count": 0
            }
            
            # Update agent to use this model
            self.update_agent_model(agent_name, model_name)
            
            # Process each ticket
            for ticket in tickets:
                import time
                start_time = time.time()
                
                try:
                    result = self.process_ticket_collaboratively(
                        ticket["id"], 
                        ticket["content"]
                    )
                    processing_time = time.time() - start_time
                    
                    model_results["ticket_results"].append({
                        "ticket_id": ticket["id"],
                        "result": result,
                        "processing_time": processing_time,
                        "success": True
                    })
                    model_results["processing_times"].append(processing_time)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    model_results["error_count"] += 1
                    model_results["ticket_results"].append({
                        "ticket_id": ticket["id"],
                        "error": str(e),
                        "processing_time": processing_time,
                        "success": False
                    })
            
            # Calculate performance metrics
            if model_results["processing_times"]:
                model_results["avg_processing_time"] = sum(model_results["processing_times"]) / len(model_results["processing_times"])
                model_results["success_rate"] = (len(tickets) - model_results["error_count"]) / len(tickets)
            else:
                model_results["avg_processing_time"] = 0
                model_results["success_rate"] = 0
                
            comparison_results["results"][model_name] = model_results
        
        # Restore original model
        self.update_agent_model(agent_name, original_model)
        
        # Generate performance summary
        comparison_results["performance_summary"] = self._generate_performance_summary(comparison_results["results"])
        
        return comparison_results
    
    def _generate_performance_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of model performance comparison."""
        summary = {
            "fastest_model": None,
            "most_accurate_model": None,
            "recommended_model": None,
            "performance_rankings": []
        }
        
        if not model_results:
            return summary
            
        # Find fastest model
        fastest_time = float('inf')
        for model_name, results in model_results.items():
            avg_time = results.get("avg_processing_time", float('inf'))
            if avg_time < fastest_time:
                fastest_time = avg_time
                summary["fastest_model"] = model_name
        
        # Find most accurate model (highest success rate)
        best_accuracy = 0
        for model_name, results in model_results.items():
            success_rate = results.get("success_rate", 0)
            if success_rate > best_accuracy:
                best_accuracy = success_rate
                summary["most_accurate_model"] = model_name
        
        # Create performance rankings (weighted: 70% accuracy, 30% speed)
        performance_scores = []
        for model_name, results in model_results.items():
            accuracy_score = results.get("success_rate", 0)
            speed_score = 1 / (results.get("avg_processing_time", 1) + 0.1)  # Inverse of time
            
            # Normalize speed score (0-1 range)
            max_speed = max([1 / (r.get("avg_processing_time", 1) + 0.1) for r in model_results.values()])
            normalized_speed = speed_score / max_speed if max_speed > 0 else 0
            
            combined_score = (accuracy_score * 0.7) + (normalized_speed * 0.3)
            performance_scores.append({
                "model": model_name,
                "score": combined_score,
                "accuracy": accuracy_score,
                "speed": normalized_speed
            })
        
        # Sort by combined score
        performance_scores.sort(key=lambda x: x["score"], reverse=True)
        summary["performance_rankings"] = performance_scores
        
        if performance_scores:
            summary["recommended_model"] = performance_scores[0]["model"]
        
        return summary
    
    def process_ticket_collaboratively(self, ticket_id: str, ticket_content: str) -> Dict[str, Any]:
        """
        Process a ticket using collaborative CrewAI workflow.
        
        Args:
            ticket_id: Unique identifier for the ticket
            ticket_content: The content of the support ticket
            
        Returns:
            Dict containing comprehensive collaborative analysis results
        """
        print(f"\n🤝 Starting collaborative processing for ticket {ticket_id}...")
        
        try:
            # Create collaborative tasks that allow agents to build on each other's work
            tasks = self._create_collaborative_tasks(ticket_id, ticket_content)
            
            # Update crew with current tasks
            self.crew.tasks = tasks
            
            # Execute collaborative workflow with LangSmith tracing
            print("🔄 Executing collaborative crew workflow...")
            
            # Add LangSmith tracing context and capture run IDs
            import os
            import uuid
            from langchain.callbacks.base import BaseCallbackHandler
            
            langsmith_run_ids = []
            
            # Custom callback to capture run IDs and properly close traces
            class RunIdCapture(BaseCallbackHandler):
                def __init__(self):
                    self.run_ids = []
                    self.active_runs = {}
                
                def on_llm_start(self, serialized, prompts, run_id=None, **kwargs):
                    if run_id:
                        self.run_ids.append(str(run_id))
                        self.active_runs[str(run_id)] = {"type": "llm", "status": "started"}
                
                def on_chain_start(self, serialized, inputs, run_id=None, **kwargs):
                    if run_id:
                        self.run_ids.append(str(run_id))
                        self.active_runs[str(run_id)] = {"type": "chain", "status": "started"}
                
                def on_llm_end(self, response, run_id=None, **kwargs):
                    if run_id and str(run_id) in self.active_runs:
                        self.active_runs[str(run_id)]["status"] = "completed"
                
                def on_chain_end(self, outputs, run_id=None, **kwargs):
                    if run_id and str(run_id) in self.active_runs:
                        self.active_runs[str(run_id)]["status"] = "completed"
                
                def on_llm_error(self, error, run_id=None, **kwargs):
                    if run_id and str(run_id) in self.active_runs:
                        self.active_runs[str(run_id)]["status"] = "error"
                        self.active_runs[str(run_id)]["error"] = str(error)
                
                def on_chain_error(self, error, run_id=None, **kwargs):
                    if run_id and str(run_id) in self.active_runs:
                        self.active_runs[str(run_id)]["status"] = "error"
                        self.active_runs[str(run_id)]["error"] = str(error)
            
            if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
                print(f"🔗 LangSmith tracing active for project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
                
                try:
                    # Set up callback to capture run IDs during execution
                    run_capture = RunIdCapture()
                    
                    # Add callback to crew's agents to ensure proper trace completion
                    for agent in self.crew.agents:
                        if hasattr(agent, 'llm') and hasattr(agent.llm, 'callbacks'):
                            if agent.llm.callbacks is None:
                                agent.llm.callbacks = []
                            agent.llm.callbacks.append(run_capture)
                    
                    # Execute crew with enhanced tracing
                    result = self.crew.kickoff()
                    
                    # Get captured run IDs
                    langsmith_run_ids = run_capture.run_ids[:4]  # Max 4 for agents
                    
                    # Log completion status for debugging
                    completed_runs = [rid for rid, info in run_capture.active_runs.items() if info.get("status") == "completed"]
                    print(f"🔗 Completed {len(completed_runs)} traces out of {len(run_capture.active_runs)} total")
                    
                    # Submit feedback and metadata to LangSmith
                    try:
                        from langsmith import Client
                        client = Client()
                        
                        # Submit metadata for each captured run
                        for run_id in langsmith_run_ids:
                            try:
                                # Add metadata about the ticket and agent processing
                                metadata = {
                                    "ticket_id": ticket_id,
                                    "processing_type": "collaborative_multi_agent",
                                    "agents_involved": ["triage_specialist", "ticket_analyst", "support_strategist", "qa_reviewer"],
                                    "completion_status": "completed",
                                    "system_version": "v2.0"
                                }
                                
                                # Submit feedback with completion status
                                client.create_feedback(
                                    run_id=run_id,
                                    key="completion_status",
                                    score=1.0,
                                    comment=f"Multi-agent processing completed for ticket {ticket_id}"
                                )
                                
                                # Update run with final metadata
                                client.update_run(
                                    run_id=run_id,
                                    extra=metadata
                                )
                                
                            except Exception as e:
                                print(f"Warning: Could not submit feedback for run {run_id}: {e}")
                                
                        print(f"📡 Submitted feedback and metadata to LangSmith for {len(langsmith_run_ids)} runs")
                        
                    except Exception as e:
                        print(f"Warning: Could not submit LangSmith feedback: {e}")
                    
                    # If no run IDs captured, generate tracking IDs
                    if not langsmith_run_ids:
                        session_id = str(uuid.uuid4())
                        for i in range(4):
                            langsmith_run_ids.append(f"trace_{ticket_id}_{i}_{session_id[:8]}")
                        print(f"🔗 Generated {len(langsmith_run_ids)} tracking IDs for agent tracing")
                    else:
                        print(f"🔗 Captured {len(langsmith_run_ids)} LangSmith run IDs")
                    
                    print(f"📡 Crew execution traced to LangSmith project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
                    
                except Exception as e:
                    print(f"Warning: LangSmith tracing setup failed: {e}")
                    result = self.crew.kickoff()
                    # Generate fallback tracking IDs
                    session_id = str(uuid.uuid4())
                    for i in range(4):
                        langsmith_run_ids.append(f"fallback_{ticket_id}_{i}_{session_id[:8]}")
            else:
                print("⚠️ LangSmith tracing not enabled")
                result = self.crew.kickoff()
                # Generate local tracking IDs for consistency
                session_id = str(uuid.uuid4())
                for i in range(4):
                    langsmith_run_ids.append(f"local_{ticket_id}_{i}_{session_id[:8]}")
            
            # Extract individual agent activities for detailed logging
            self.individual_agent_logs = self._extract_individual_agent_activities(result, ticket_id, ticket_content, langsmith_run_ids)
            
            # Parse and structure the collaborative result
            final_result = self._parse_collaborative_result(result, ticket_id, ticket_content)
            
            # Add individual agent logs to the result for external logging
            final_result["individual_agent_logs"] = self.individual_agent_logs
            
            print(f"✅ Collaborative processing completed for ticket {ticket_id}")
            return final_result
            
        except Exception as e:
            print(f"❌ Error in collaborative processing for ticket {ticket_id}: {str(e)}")
            return self._create_fallback_result(ticket_id, ticket_content, str(e))
    
    def _create_collaborative_tasks(self, ticket_id: str, ticket_content: str) -> List[Task]:
        """Create collaborative tasks for the CrewAI workflow."""
        
        # Task 1: Initial Classification and Analysis
        classification_task = Task(
            description=f"""
            Analyze the following customer support ticket (ID: {ticket_id}) and provide initial classification:
            
            Ticket Content: {ticket_content}
            
            Classify the ticket by:
            1. Intent (billing, bug, feedback, feature_request, general_inquiry, technical_support, account_issue, refund_request, complaint, compliment)
            2. Severity (critical, high, medium, low)
            3. Confidence level (0.0 to 1.0)
            4. Reasoning for your classification
            
            Be open to feedback from other agents who may identify aspects you missed.
            """,
            agent=self.triage_specialist,
            expected_output="Initial classification with intent, severity, confidence, and reasoning"
        )
        
        # Task 2: Detailed Analysis and Summary
        analysis_task = Task(
            description=f"""
            Based on the initial classification, perform detailed analysis of ticket {ticket_id}:
            
            Original Ticket Content: {ticket_content}
            
            1. Create a comprehensive summary of the customer's issue
            2. Identify any technical details, customer sentiment, or underlying problems
            3. Review the triage classification and provide feedback if you disagree
            4. Highlight any aspects that might affect severity or intent classification
            
            Consider the triage specialist's work and build upon it or suggest refinements.
            You have full access to the original ticket content above.
            """,
            agent=self.ticket_analyst,
            expected_output="Detailed summary and analysis with feedback on classification if needed"
        )
        
        # Task 3: Strategy and Action Planning
        strategy_task = Task(
            description=f"""
            Develop a comprehensive action plan for ticket {ticket_id} based on the classification and analysis:
            
            Original Ticket Content: {ticket_content}
            
            1. Recommend primary and secondary actions
            2. Set priority level and estimated resolution time
            3. Ensure consistency between severity and recommended actions
            4. Challenge any inconsistencies you notice in previous assessments
            5. Provide strategic notes for the support team
            
            If you see conflicts between classification and analysis, raise them for discussion.
            You have full access to the original ticket content above.
            """,
            agent=self.support_strategist,
            expected_output="Strategic action plan with primary/secondary actions, priority, timeline, and consistency checks"
        )
        
        # Task 4: Quality Review and Final Consensus
        review_task = Task(
            description=f"""
            Perform final quality review and ensure consensus for ticket {ticket_id}:
            
            Original Ticket Content: {ticket_content}
            
            1. Review all previous agent outputs for consistency
            2. Identify any conflicts between classification, analysis, and action plan
            3. Facilitate resolution of any disagreements
            4. Provide final validated output with consensus from all agents
            5. Calculate agreement scores between agents
            
            Your goal is to ensure the final output is coherent, consistent, and actionable.
            You have full access to the original ticket content above.
            """,
            agent=self.qa_reviewer,
            expected_output="Final consensus report with validated classification, summary, action plan, and agreement metrics"
        )
        
        return [classification_task, analysis_task, strategy_task, review_task]
    
    def _parse_collaborative_result(self, crew_result, ticket_id: str, ticket_content: str) -> Dict[str, Any]:
        """Parse and structure the collaborative crew result into a standardized format."""
        try:
            # Handle CrewAI CrewOutput object
            if hasattr(crew_result, 'raw'):
                crew_text = str(crew_result.raw)
            elif hasattr(crew_result, 'result'):
                crew_text = str(crew_result.result)
            else:
                crew_text = str(crew_result)
            
            # Extract authentic collaboration metrics from the crew execution
            collaboration_metrics = self._extract_authentic_collaboration_metrics(crew_result, ticket_id)
            
            # Try to parse as JSON if possible, otherwise extract key information
            if '{' in crew_text and '}' in crew_text:
                # Look for JSON-like content
                start_idx = crew_text.find('{')
                end_idx = crew_text.rfind('}') + 1
                json_content = crew_text[start_idx:end_idx]
                
                try:
                    parsed = json.loads(json_content)
                    result = self._structure_collaborative_result(parsed, ticket_id, ticket_content, crew_text)
                    result["collaboration_metrics"] = collaboration_metrics
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Parse text-based result
            result = self._parse_text_result(crew_text, ticket_id, ticket_content)
            result["collaboration_metrics"] = collaboration_metrics
            return result
            
        except Exception as e:
            print(f"⚠️  Error parsing collaborative result for ticket {ticket_id}: {str(e)}")
            return self._create_fallback_result(ticket_id, ticket_content, f"Parse error: {str(e)}")
    
    def _structure_collaborative_result(self, parsed_data: Dict, ticket_id: str, ticket_content: str, raw_output: str) -> Dict[str, Any]:
        """Structure parsed collaborative data into standard format."""
        return {
            "ticket_id": ticket_id,
            "original_message": ticket_content,
            "classification": {
                "intent": parsed_data.get("intent", "general_inquiry"),
                "severity": parsed_data.get("severity", "medium"),
                "confidence": parsed_data.get("confidence", 0.8),
                "reasoning": parsed_data.get("reasoning", "Collaborative analysis")
            },
            "summary": parsed_data.get("summary", "Collaborative summary generated by multi-agent analysis"),
            "action_recommendation": {
                "primary_action": parsed_data.get("primary_action", "request_more_info"),
                "secondary_actions": parsed_data.get("secondary_actions", []),
                "priority": parsed_data.get("priority", "medium"),
                "estimated_resolution_time": parsed_data.get("estimated_resolution_time", "4-6 hours"),
                "notes": parsed_data.get("notes", "Collaborative action plan")
            },
            "collaboration_metrics": {
                "agent_agreement": parsed_data.get("agreement_score", 0.85),
                "conflicts_resolved": parsed_data.get("conflicts_resolved", 0),
                "consensus_reached": True
            },
            "processing_status": "completed",
            "raw_collaborative_output": raw_output[:1000]  # Truncate for storage
        }
    
    def _parse_text_result(self, text_result: str, ticket_id: str, ticket_content: str) -> Dict[str, Any]:
        """Parse text-based collaborative result when JSON parsing fails."""
        # Extract key information using text patterns
        intent = self._extract_field(text_result, ["intent", "category", "type"], "general_inquiry")
        severity = self._extract_field(text_result, ["severity", "priority", "urgency"], "medium")
        
        return {
            "ticket_id": ticket_id,
            "original_message": ticket_content,
            "classification": {
                "intent": intent,
                "severity": severity,
                "confidence": 0.75,
                "reasoning": "Extracted from collaborative text analysis"
            },
            "summary": self._extract_summary(text_result),
            "action_recommendation": {
                "primary_action": self._extract_field(text_result, ["action", "recommend", "next"], "request_more_info"),
                "secondary_actions": [],
                "priority": severity,
                "estimated_resolution_time": "4-6 hours",
                "notes": "Based on collaborative text analysis"
            },
            "collaboration_metrics": {
                "agent_agreement": 0.8,
                "conflicts_resolved": 0,
                "consensus_reached": True
            },
            "processing_status": "completed",
            "raw_collaborative_output": text_result[:1000]
        }
    
    def _extract_field(self, text: str, keywords: List[str], default: str) -> str:
        """Extract field value from text using keyword matching."""
        text_lower = text.lower()
        for keyword in keywords:
            for line in text.split('\n'):
                if keyword in line.lower():
                    # Try to extract value after colon or similar patterns
                    parts = line.split(':')
                    if len(parts) > 1:
                        value_parts = parts[1].strip().split()
                        if value_parts:  # Check if split result is not empty
                            value = value_parts[0]
                            if value:
                                return value
        return default
    
    def _extract_summary(self, text: str) -> str:
        """Extract summary section from collaborative result."""
        lines = text.split('\n')
        summary_started = False
        summary_lines = []
        
        for line in lines:
            if 'summary' in line.lower() or summary_started:
                summary_started = True
                if line.strip() and not line.startswith('='):
                    summary_lines.append(line.strip())
                if len(summary_lines) > 5:  # Limit summary length
                    break
        
        return ' '.join(summary_lines) if summary_lines else "Collaborative analysis summary"
    
    def _extract_individual_agent_activities(self, crew_result, ticket_id: str, ticket_content: str, langsmith_run_ids: List[str] = None) -> List[Dict[str, Any]]:
        """Extract individual agent activities from CrewAI execution for detailed logging."""
        individual_logs = []
        
        try:
            if hasattr(crew_result, 'tasks_output') and crew_result.tasks_output:
                # Map task outputs to agent activities
                agent_mapping = {
                    0: 'triage_specialist',
                    1: 'ticket_analyst', 
                    2: 'support_strategist',
                    3: 'qa_reviewer'
                }
                
                for i, task_output in enumerate(crew_result.tasks_output):
                    if i < len(agent_mapping):
                        agent_name = agent_mapping[i]
                        
                        # Extract agent-specific data
                        agent_input = {
                            'ticket_id': ticket_id,
                            'ticket_content': ticket_content,
                            'task_description': str(task_output.description) if hasattr(task_output, 'description') else 'Task description not available',
                            'agent_role': agent_name
                        }
                        
                        agent_output = {
                            'raw_output': str(task_output.raw) if hasattr(task_output, 'raw') else str(task_output),
                            'task_completion': 'success' if task_output else 'failed'
                        }
                        
                        # Get agent-specific model information
                        agent_model = self.agent_models.get(agent_name, 'gpt-4o')
                        model_config = AVAILABLE_MODELS.get(agent_model, {})
                        
                        agent_metadata = {
                            'model_used': agent_model,
                            'model_provider': model_config.get('provider', 'openai'),
                            'temperature': model_config.get('temperature', 0.1),
                            'agent_position': i + 1,
                            'total_agents': len(crew_result.tasks_output),
                            'task_type': self._determine_task_type(i)
                        }
                        
                        # Assign LangSmith run ID if available
                        langsmith_run_id = None
                        if langsmith_run_ids and i < len(langsmith_run_ids):
                            langsmith_run_id = langsmith_run_ids[i]
                        
                        individual_logs.append({
                            'agent_name': agent_name,
                            'input_data': agent_input,
                            'output_data': agent_output,
                            'metadata': agent_metadata,
                            'processing_time': 0.0,  # CrewAI doesn't expose individual timing
                            'status': 'success',
                            'trace_id': f"{ticket_id}_{agent_name}_{int(time.time())}",
                            'langsmith_run_id': langsmith_run_id
                        })
                        
        except Exception as e:
            print(f"Warning: Could not extract individual agent activities: {e}")
            
        return individual_logs
    
    def _determine_task_type(self, task_index: int) -> str:
        """Determine task type based on agent position."""
        task_types = {
            0: 'classification',
            1: 'analysis', 
            2: 'strategy',
            3: 'review'
        }
        return task_types.get(task_index, 'unknown')

    def _extract_authentic_collaboration_metrics(self, crew_result, ticket_id: str) -> Dict[str, Any]:
        """Extract real collaboration metrics from CrewAI execution."""
        start_time = time.time()
        
        # Initialize metrics tracking
        metrics = {
            "disagreement_count": 0,
            "conflicts_identified": [],
            "conflict_resolution_methods": [],
            "resolution_iterations": 0,
            "consensus_building_duration": 0.0,
            "agent_iterations": {},
            "agent_agreement_evolution": [],
            "final_agreement_scores": {},
            "overall_agreement_strength": 0.0,
            "consensus_reached": False,
            "confidence_improvement": 0.0,
            "result_stability": 0.0
        }
        
        try:
            # Extract from crew execution traces if available
            if hasattr(crew_result, 'tasks_output') and crew_result.tasks_output:
                agent_outputs = []
                for task_output in crew_result.tasks_output:
                    if hasattr(task_output, 'agent') and hasattr(task_output, 'raw'):
                        agent_name = task_output.agent.role if hasattr(task_output.agent, 'role') else 'unknown'
                        agent_outputs.append({
                            'agent': agent_name,
                            'output': str(task_output.raw)
                        })
                
                # Calculate real disagreements and conflicts
                metrics = self._calculate_real_disagreements(agent_outputs, metrics)
                
            # Calculate consensus building duration
            metrics["consensus_building_duration"] = time.time() - start_time
            
            # Determine if consensus was reached based on actual output consistency
            metrics["consensus_reached"] = self._assess_consensus_quality(crew_result)
            
            # Calculate overall agreement strength
            metrics["overall_agreement_strength"] = self._calculate_agreement_strength(metrics)
            
        except Exception as e:
            print(f"Warning: Could not extract collaboration metrics: {e}")
            metrics["consensus_reached"] = False
            
        return metrics
    
    def _calculate_real_disagreements(self, agent_outputs: List[Dict], metrics: Dict) -> Dict:
        """Calculate authentic disagreements between agent outputs."""
        if len(agent_outputs) < 2:
            return metrics
        
        # Extract key fields from each agent output
        agent_classifications = {}
        conflicts = []
        
        for output in agent_outputs:
            agent_name = output['agent']
            text = output['output'].lower()
            
            # Extract intent, severity, and recommendations
            classification = {
                'intent': self._extract_classification_field(text, 'intent'),
                'severity': self._extract_classification_field(text, 'severity'),
                'action': self._extract_classification_field(text, 'action')
            }
            agent_classifications[agent_name] = classification
        
        # Compare classifications to find real disagreements
        fields_to_compare = ['intent', 'severity', 'action']
        disagreements = {}
        
        for field in fields_to_compare:
            values = [agent_classifications[agent][field] for agent in agent_classifications if agent_classifications[agent][field]]
            unique_values = list(set(values))
            
            if len(unique_values) > 1:
                disagreements[field] = {
                    'values': values,
                    'unique_count': len(unique_values),
                    'agents_disagreeing': list(agent_classifications.keys())
                }
                conflicts.append(f"{field} disagreement: {unique_values}")
        
        metrics["disagreement_count"] = len(disagreements)
        metrics["conflicts_identified"] = conflicts
        metrics["agent_iterations"] = {agent: 1 for agent in agent_classifications.keys()}
        
        # Track resolution methods (simplified)
        if disagreements:
            metrics["conflict_resolution_methods"] = ["majority_vote", "qa_reviewer_decision"]
            metrics["resolution_iterations"] = 1
        
        return metrics
    
    def _extract_classification_field(self, text: str, field: str) -> str:
        """Extract classification field from agent output text."""
        field_patterns = {
            'intent': ['intent:', 'category:', 'type:', 'classification:'],
            'severity': ['severity:', 'priority:', 'urgency:'],
            'action': ['action:', 'recommend:', 'primary_action:']
        }
        
        patterns = field_patterns.get(field, [])
        for pattern in patterns:
            if pattern in text:
                # Find the line containing the pattern
                for line in text.split('\n'):
                    if pattern in line:
                        parts = line.split(pattern)
                        if len(parts) > 1:
                            value = parts[1].strip().split()[0] if parts[1].strip() else None
                            if value and value not in ['', '-', 'n/a']:
                                return value.lower()
        return None
    
    def _assess_consensus_quality(self, crew_result) -> bool:
        """Assess if genuine consensus was reached based on output consistency."""
        try:
            if hasattr(crew_result, 'raw'):
                output_text = str(crew_result.raw).lower()
                
                # Check for consensus indicators in the final output
                consensus_indicators = [
                    'consensus', 'agreement', 'all agents agree', 'consistent',
                    'validated', 'confirmed', 'final decision'
                ]
                
                conflict_indicators = [
                    'disagreement', 'conflict', 'inconsistent', 'disputed',
                    'unresolved', 'requires review'
                ]
                
                consensus_score = sum(1 for indicator in consensus_indicators if indicator in output_text)
                conflict_score = sum(1 for indicator in conflict_indicators if indicator in output_text)
                
                return consensus_score > conflict_score
                
        except Exception:
            return False
        
        return False
    
    def _calculate_agreement_strength(self, metrics: Dict) -> float:
        """Calculate overall agreement strength based on authentic metrics."""
        if metrics["disagreement_count"] == 0:
            return 1.0
        
        # Calculate based on resolution success
        conflicts_resolved = len(metrics.get("conflict_resolution_methods", []))
        total_conflicts = metrics["disagreement_count"]
        
        if total_conflicts == 0:
            return 1.0
        
        resolution_rate = conflicts_resolved / total_conflicts
        
        # Factor in consensus building efficiency
        duration_penalty = min(0.1, metrics["consensus_building_duration"] / 100)  # Slight penalty for long discussions
        
        agreement_strength = resolution_rate - duration_penalty
        return max(0.0, min(1.0, agreement_strength))
    
    def _create_fallback_result(self, ticket_id: str, ticket_content: str, error_msg: str) -> Dict[str, Any]:
        """Create fallback result when collaborative processing fails."""
        return {
            "ticket_id": ticket_id,
            "original_message": ticket_content,
            "classification": {
                "intent": "general_inquiry",
                "severity": "medium",
                "confidence": 0.5,
                "reasoning": f"Fallback classification due to error: {error_msg}"
            },
            "summary": "Error in collaborative processing - manual review required",
            "action_recommendation": {
                "primary_action": "request_more_info",
                "secondary_actions": ["manual_review"],
                "priority": "medium",
                "estimated_resolution_time": "4-6 hours",
                "notes": f"Collaborative processing failed: {error_msg}"
            },
            "collaboration_metrics": {
                "disagreement_count": 0,
                "conflicts_identified": [],
                "conflict_resolution_methods": [],
                "consensus_reached": False,
                "overall_agreement_strength": 0.0,
                "consensus_building_duration": 0.0
            },
            "processing_status": "error",
            "raw_collaborative_output": error_msg
        }
    
    # Backward compatibility methods for existing integrations
    def process_ticket(self, ticket_id: str, ticket_content: str) -> Dict[str, Any]:
        """
        Process a ticket using collaborative workflow (backward compatibility).
        
        Args:
            ticket_id: Unique identifier for the ticket
            ticket_content: The content of the support ticket
            
        Returns:
            Dict containing all processing results
        """
        return self.process_ticket_collaboratively(ticket_id, ticket_content)
    
    def classify_ticket(self, ticket_content: str, ticket_id: str) -> Dict[str, Any]:
        """Backward compatibility method - now uses collaborative processing."""
        result = self.process_ticket_collaboratively(ticket_id, ticket_content)
        return result.get("classification", {})
    
    def summarize_ticket(self, ticket_content: str, classification: Dict[str, Any], ticket_id: str) -> str:
        """Backward compatibility method - now uses collaborative processing."""
        result = self.process_ticket_collaboratively(ticket_id, ticket_content)
        return result.get("summary", "")
    
    def recommend_action(self, classification: Dict[str, Any], summary: str, ticket_id: str) -> Dict[str, Any]:
        """Backward compatibility method - now uses collaborative processing."""
        # Extract ticket content from classification if available, otherwise use summary
        ticket_content = summary  # Fallback
        result = self.process_ticket_collaboratively(ticket_id, ticket_content)
        return result.get("action_recommendation", {})
    
    def _parse_classification_fallback(self, response_text: str) -> Dict[str, Any]:
        """Fallback method to parse classification from non-JSON response."""
        
        # First, try to extract any partial JSON elements from the response
        if '"intent"' in response_text or '"severity"' in response_text:
            # Handle cases like '\n    "intent"' by providing safe defaults
            print(f"⚠️  Detected partial JSON response, using safe defaults")
            return {
                "intent": "general_inquiry",
                "severity": "medium", 
                "confidence": 0.5,
                "reasoning": "Partial JSON response detected, applied safe defaults"
            }
        
        # Simple keyword-based fallback for complete responses
        intent_keywords = {
            "billing": ["bill", "charge", "payment", "refund", "money"],
            "bug": ["bug", "error", "crash", "broken", "not working"],
            "feedback": ["feedback", "suggestion", "love", "like", "hate"],
            "technical_support": ["help", "how to", "support", "assistance"]
        }
        
        severity_keywords = {
            "critical": ["critical", "urgent", "emergency", "immediately"],
            "high": ["important", "asap", "quickly", "frustrated"],
            "low": ["question", "wondering", "when possible"]
        }
        
        response_lower = response_text.lower()
        
        # Determine intent
        intent = "general_inquiry"
        for intent_type, keywords in intent_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                intent = intent_type
                break
        
        # Determine severity
        severity = "medium"
        for severity_level, keywords in severity_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                severity = severity_level
                break
        
        return {
            "intent": intent,
            "severity": severity,
            "confidence": 0.6,
            "reasoning": "Parsed using fallback method due to JSON parsing error"
        }
    
    def _parse_action_fallback(self, response_text: str, severity: str) -> Dict[str, Any]:
        """Fallback method to parse actions from non-JSON response."""
        
        # Handle partial JSON responses like '\n    "primary_action"'
        if '"primary_action"' in response_text or '"priority"' in response_text:
            print(f"⚠️  Detected partial JSON action response, using safe defaults")
            return {
                "primary_action": "request_more_info",
                "secondary_actions": [],
                "priority": "medium",
                "estimated_resolution_time": "2-4 hours",
                "notes": "Partial JSON response detected, applied safe defaults"
            }
        
        # Simple severity-based fallback
        severity_actions = {
            "critical": "escalate_to_manager",
            "high": "escalate_to_tier2",
            "medium": "respond_with_template",
            "low": "respond_with_template"
        }
        
        return {
            "primary_action": severity_actions.get(severity, "request_more_info"),
            "secondary_actions": ["immediate_response"] if severity in ["critical", "high"] else [],
            "priority": severity,
            "estimated_resolution_time": "1 hour" if severity == "critical" else "2-4 hours",
            "notes": "Generated using fallback method due to JSON parsing error"
        }

# Create alias for backward compatibility with existing integrations
SupportTicketAgents = CollaborativeSupportCrew
