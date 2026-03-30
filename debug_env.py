"""
Deep Diagnostic Script for CrewAI & Gemini API Authentication
"""
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_core.language_models.llms import BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent

def run_diagnostics():
    print("--- 1. Environment Check ---")
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        masked_key = "*" * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else "****"
        print(f"GOOGLE_API_KEY loaded: {masked_key}")
    else:
        print("GOOGLE_API_KEY is MISSING from the environment.")

    print("\n--- 2. Class Inspection ---")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
        print("ChatGoogleGenerativeAI instantiation: SUCCESS")
        
        is_base_llm = isinstance(llm, BaseLLM)
        print(f"isinstance(llm, BaseLLM): {is_base_llm}")
        
        print(f"\nMRO (Method Resolution Order) for type(llm):")
        for idx, cls in enumerate(type(llm).mro()):
            print(f"  {idx}. {cls}")
            
    except Exception as e:
        print(f"Failed to instantiate LLM: {e}")
        return

    print("\n--- 3. Minimal Agent Test ---")
    try:
        agent = Agent(
            role="test",
            goal="test",
            backstory="test",
            llm=llm
        )
        print("Agent strictly instantiated without validation errors.")
    except Exception as e:
        print("Agent Instantiation Exception Caught!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details:\n{e}")

if __name__ == "__main__":
    run_diagnostics()
