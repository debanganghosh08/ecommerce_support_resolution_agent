"""
CrewAI tasks for E-commerce Support Resolution.
"""
from crewai import Task
from src.schemas import EcommerceSupportResponse

def create_triage_task(agent, ticket_text: str, order_context: dict) -> Task:
    return Task(
        description=(
            f"Analyze the following incoming customer ticket and associated order context.\n\n"
            f"Ticket: {ticket_text}\n"
            f"Order Context: {order_context}\n\n"
            "Identify the core issue classification. Then, review the order context to point "
            "out any missing details that might be required to resolve the issue."
        ),
        expected_output=(
            "A summary classifying the issue type and listing any missing context "
            "necessary for resolution."
        ),
        agent=agent
    )

def create_retriever_task(agent) -> Task:
    return Task(
        description=(
            "Based on the output from the Triage Specialist regarding the customer's core issue, "
            "formulate highly specific search queries. Use your policy_search_tool to find the exact "
            "policy rules regarding the issue. Return the exact excerpts alongside their citations."
        ),
        expected_output=(
            "A comprehensive list of verbatim policy excerpts and strict citations "
            "relevant to resolving the ticket."
        ),
        agent=agent
    )

def create_writer_task(agent) -> Task:
    return Task(
        description=(
            "Using ONLY the retrieved policy excerpts and strict citations from the Policy Retriever, "
            "draft an empathetic, professional response to the customer regarding their issue. "
            "Also document the internal rationale for why the decision was made based on the policy."
        ),
        expected_output=(
            "A draft message to the customer, accompanied by the rationale "
            "and citations used to form it."
        ),
        agent=agent
    )

def create_compliance_task(agent) -> Task:
    return Task(
        description=(
            "Vigorously audit the draft response against the original retrieved facts. Success criteria: "
            "'Zero unsupported claims.' If any claim in the rationale or drafted response lacks a direct "
            "and explicit citation from the retrieved context, you MUST change the decision to 'Escalate'.\n"
            "Format the final output exactly according to the provided schema."
        ),
        expected_output="A fully validated JSON structure matching the EcommerceSupportResponse schema.",
        agent=agent,
        output_pydantic=EcommerceSupportResponse
    )
