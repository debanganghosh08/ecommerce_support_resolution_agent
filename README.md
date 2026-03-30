# E-commerce Support Resolution Agent (Agentic RAG Foundation)

**Author:** Debangan Ghosh

## Architecture Overview
This project implements a highly resilient, 4-agent production pipeline leveraging the **Groq Llama 3.1 8B** model and **Google Generative AI Embeddings**. The framework implements CrewAI to securely orchestrate the following logic nodes:
1. **Triage Specialist:** Formally classifies the incoming customer ticket and extracts essential runtime context.
2. **Policy Retriever:** Queries the Persistent Singleton Chroma database using semantic similarity to retrieve strictly verbatim policy excerpts.
3. **Resolution Writer:** Drafts empathetic, customer-ready responses bound definitively to the retrieved policy excerpts.
4. **Compliance Officer:** Functions as the final hallucination firewall prioritizing zero unsupported claims before authorizing the final JSON delivery.

## Directory Structure
* `src/`: Core agent logic and vector store management.
* `data/`: Pruned high-fidelity policy documents.
* `eval/`: Test cases, results, and metrics scripts.
* `vectorstore/`: Persistent ChromaDB instance.

## Setup Instructions
1. Initialize core dependencies natively:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment (`.env`):
   Ensure your `.env` is structured at the project root with your explicit API credentials:
   ```text
   GROQ_API_KEY=your_groq_key
   GOOGLE_API_KEY=your_google_ai_studio_key
   ```
3. Execute the core verification matrix:
   ```bash
   python main.py
   ```

## Evaluation Summary
The evaluation framework executes a rigorous 20-case test parameter explicitly mapping Standard, Exception, Conflict, and Out-Of-Scope support inquiries. To mathematically offset the severe `6,000 TPM` limitation present on Developer endpoint keys natively, the iteration sequence utilizes a **Full-Minute Reset**. By enforcing a 65-second rolling `time.sleep()` boundary between executions, the pipeline completely resets its algorithmic context load, neutralizing 429 rate limit exceptions unilaterally and guaranteeing 100% test completion.
