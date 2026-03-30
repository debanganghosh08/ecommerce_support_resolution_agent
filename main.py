"""
Main entry point for Agentic RAG Foundation.
"""
import os
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded immediately before importing modules that need them
load_dotenv()

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["GEMINI_API_VERSION"] = "v1beta"

from crewai import Crew, Process

from src.agents import (
    get_triage_agent,
    get_policy_retriever_agent,
    get_resolution_writer_agent,
    get_compliance_agent
)
from src.tasks import (
    create_triage_task,
    create_retriever_task,
    create_writer_task,
    create_compliance_task
)

def run_ecommerce_support_crew(ticket_text: str, order_context_json: str):
    """
    Orchestrates the resolution pipeline sequentially.
    """
    # Parse the incoming JSON context
    order_context = json.loads(order_context_json)
    
    # Instantiate agents
    triage_agent = get_triage_agent()
    retriever_agent = get_policy_retriever_agent()
    writer_agent = get_resolution_writer_agent()
    compliance_agent = get_compliance_agent()
    
    # Instantiate sequential tasks
    triage_task = create_triage_task(triage_agent, ticket_text, order_context)
    retriever_task = create_retriever_task(retriever_agent)
    writer_task = create_writer_task(writer_agent)
    compliance_task = create_compliance_task(compliance_agent)
    
    # Assemble crew with a strict sequential process
    crew = Crew(
        agents=[triage_agent, retriever_agent, writer_agent, compliance_agent],
        tasks=[triage_task, retriever_task, writer_task, compliance_task],
        process=Process.sequential,
        verbose=True
    )
    
    print(f"Starting support resolution for ticket: '{ticket_text}'...")
    result = crew.kickoff()
    return result

if __name__ == "__main__":    
    if "GOOGLE_API_KEY" not in os.environ:
        print("Warning: GOOGLE_API_KEY not found in environment. CrewAI interactions with Gemini will fail.")
    
    print("\nFINAL SUBMISSION CONFIGURATION: Balanced Groq 8B Matrix Engaged.")

    # Small Example Run using the 'melted cookies' case
    print("\n--- Example Run: Melted Cookies Validation ---")
    
    sample_ticket = "My order arrived late and the cookies are entirely melted! I want a refund."
    sample_context = {
        "order_date": "2026-03-24",
        "delivery_date": "2026-03-28",
        "item_category": "perishables",
        "fulfillment_type": "1st party",
        "shipping_region": "Arizona",
        "order_status": "delivered_late"
    }
    
    print("Executing Crew Process (Sequential)...\n")
    try:
        final_output = run_ecommerce_support_crew(sample_ticket, json.dumps(sample_context))
        
        print("\n--- FINAL COMPLIANCE AGENT OUTPUT (Pydantic JSON) ---")
        try:
            # If output is a valid Pydantic object
            if hasattr(final_output, "model_dump_json"):
                print(final_output.model_dump_json(indent=2))
            # Or if CrewAI wrapped it in its own strict Output object
            elif hasattr(final_output, "pydantic") and final_output.pydantic:
                print(final_output.pydantic.model_dump_json(indent=2))
            else:
                print(final_output)
        except Exception as display_err:
            print("Could not parse CrewAI output to JSON directly:\n", final_output)
            print("Display format error:", display_err)
            
    except Exception as e:
        print(f"Failed to execute Crew pipeline: {e}")
