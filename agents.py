"""
CrewAI collaborative agent definitions for customer support ticket processing.
Implements a multi-agent system where agents collaborate and refine each other's work.
"""

import json
from typing import Dict, Any, List
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from config import LLM_MODEL, OPENAI_API_KEY

class CollaborativeSupportCrew:
    """
    CrewAI collaborative system for customer support ticket processing.
    Agents work together, provide feedback, and refine each other's outputs.
    """
    
    def __init__(self):
        """Initialize the collaborative crew with specialized agents."""
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1  # Low temperature for consistent results
        )
        
        # Initialize collaborative agents
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
            llm=self.llm,
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
            llm=self.llm,
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
            llm=self.llm,
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
            llm=self.llm,
            max_execution_time=300
        )
    
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
            
            # Execute collaborative workflow
            print(f"🔄 Executing collaborative crew workflow...")
            result = self.crew.kickoff()
            
            # Parse and structure the collaborative result
            final_result = self._parse_collaborative_result(result, ticket_id, ticket_content)
            
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
            
            1. Create a comprehensive summary of the customer's issue
            2. Identify any technical details, customer sentiment, or underlying problems
            3. Review the triage classification and provide feedback if you disagree
            4. Highlight any aspects that might affect severity or intent classification
            
            Consider the triage specialist's work and build upon it or suggest refinements.
            """,
            agent=self.ticket_analyst,
            expected_output="Detailed summary and analysis with feedback on classification if needed"
        )
        
        # Task 3: Strategy and Action Planning
        strategy_task = Task(
            description=f"""
            Develop a comprehensive action plan for ticket {ticket_id} based on the classification and analysis:
            
            1. Recommend primary and secondary actions
            2. Set priority level and estimated resolution time
            3. Ensure consistency between severity and recommended actions
            4. Challenge any inconsistencies you notice in previous assessments
            5. Provide strategic notes for the support team
            
            If you see conflicts between classification and analysis, raise them for discussion.
            """,
            agent=self.support_strategist,
            expected_output="Strategic action plan with primary/secondary actions, priority, timeline, and consistency checks"
        )
        
        # Task 4: Quality Review and Final Consensus
        review_task = Task(
            description=f"""
            Perform final quality review and ensure consensus for ticket {ticket_id}:
            
            1. Review all previous agent outputs for consistency
            2. Identify any conflicts between classification, analysis, and action plan
            3. Facilitate resolution of any disagreements
            4. Provide final validated output with consensus from all agents
            5. Calculate agreement scores between agents
            
            Your goal is to ensure the final output is coherent, consistent, and actionable.
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
            
            # Try to parse as JSON if possible, otherwise extract key information
            if '{' in crew_text and '}' in crew_text:
                # Look for JSON-like content
                start_idx = crew_text.find('{')
                end_idx = crew_text.rfind('}') + 1
                json_content = crew_text[start_idx:end_idx]
                
                try:
                    parsed = json.loads(json_content)
                    return self._structure_collaborative_result(parsed, ticket_id, ticket_content, crew_text)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: Parse text-based result
            return self._parse_text_result(crew_text, ticket_id, ticket_content)
            
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
                        value = parts[1].strip().split()[0]
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
                "agent_agreement": 0.0,
                "conflicts_resolved": 0,
                "consensus_reached": False
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
