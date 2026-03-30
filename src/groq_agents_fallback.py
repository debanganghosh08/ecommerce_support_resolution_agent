"""
Groq Fallback CrewAI agents for the E-commerce Support Resolution.
"""
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

from crewai import Agent, LLM
from src.vector_store import PolicySearchTool

# Fallback: Pivot to Llama 3.1 70B via Groq
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

def get_triage_agent() -> Agent:
    return Agent(
        role="Triage Specialist",
        goal="Classify the incoming customer ticket and identify any missing required order context.",
        backstory=(
            "You are the first point of contact for customer support. Your job is to read the "
            "incoming ticket, classify the core issue (e.g., Refund, Shipping, Fraud), and check "
            "if vital order context like order_date, shipping_region, etc. is provided."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def get_policy_retriever_agent() -> Agent:
    return Agent(
        role="Policy Retriever",
        goal="Search the vector store using the triage context and return verbatim policy excerpts with strict citations.",
        backstory=(
            "You are an expert at searching corporate knowledge bases. You MUST use the PolicySearchTool "
            "to find exact policy clauses relevant to the customer's issue. You only return the verbatim excerpts "
            "and their accompanying [URL + Section/Chunk ID] citations. You never invent policy."
        ),
        tools=[PolicySearchTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def get_resolution_writer_agent() -> Agent:
    return Agent(
        role="Resolution Writer",
        goal="Draft a customer-ready response based ONLY on retrieved evidence from the Policy Retriever.",
        backstory=(
            "You draft empathetic, professional responses to customers. However, you are strictly bound "
            "by the retrieved policy evidence. You MUST NOT make unsupported claims or promise actions "
            "that are not explicitly backed by the citations provided in your context."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def get_compliance_agent() -> Agent:
    return Agent(
        role="Compliance and Safety Officer",
        goal=(
            "Ensure the Resolution Writer's draft strictly adheres to retrieved policy and contains zero "
            "unsupported claims. Output the final structured schema."
        ),
        backstory=(
            "You are the final authoritative gatekeeper. Your success criteria is strictly 'Zero unsupported claims.' "
            "Compare every claim in the Resolution Rationale against the provided Policy Excerpts. "
            "CRITICAL: If a claim exists without an explicitly matching citation, the task must fail and you MUST "
            "return an 'Escalate' status in the decision. DO NOT approve a response without solid citation backing."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
