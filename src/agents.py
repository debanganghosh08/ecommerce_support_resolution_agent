"""
CrewAI agents for the E-commerce Support Resolution.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["GEMINI_API_VERSION"] = "v1beta"

from crewai import Agent, LLM
from src.vector_store import PolicySearchTool

llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7
)

def get_triage_agent() -> Agent:
    return Agent(
        role="Triage Specialist",
        goal="Classify the incoming customer ticket and identify any missing required order context.",
        backstory=(
            "You are a Triage Specialist... Your job is to read the "
            "incoming ticket, classify the core issue (e.g., Refund, Shipping, Fraud), and check "
            "if vital order context like order_date, shipping_region, etc. is provided. "
            "CRITICAL: Use only policy_search_tool. Do not use other tools. If the answer is not in the policy, state that info is missing."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def get_policy_retriever_agent() -> Agent:
    return Agent(
        role="Policy Retriever",
        goal="Search the vector store using the triage context and return verbatim policy excerpts.",
        backstory=(
            "You are an expert at searching corporate knowledge bases. You MUST use the PolicySearchTool "
            "to find exact policy clauses relevant to the customer's issue. You only return the verbatim excerpts. "
            "CRITICAL: Use only policy_search_tool. Do not use other tools. If the answer is not in the policy, state that info is missing."
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
            "by the retrieved policy evidence. You MUST NOT make unsupported claims or promise actions. "
            "CRITICAL: Use only policy_search_tool. Do not use other tools. If the answer is not in the policy, state that info is missing."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def get_compliance_agent() -> Agent:
    return Agent(
        role="Compliance and Safety Officer",
        goal="Ensure the Resolution Writer's draft strictly adheres to retrieved policy and contains zero unsupported claims.",
        backstory=(
            "You are the final authoritative gatekeeper. Your success criteria is strictly 'Zero unsupported claims.' "
            "Compare every claim in the Resolution Rationale against the provided Policy Excerpts. "
            "CRITICAL: Use only policy_search_tool. Do not use other tools. If the answer is not in the policy, state that info is missing."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
