"""
Vector Store operations for Agentic RAG Foundation.
"""
import os
import glob
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crewai.tools import BaseTool

from src.ingestion import DocumentIngestor
import src.utils as utils

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vectorstore")

class VectorStoreManager:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY") # Explicit pass
        )
        self.vector_store = Chroma(
            collection_name="policy_corpus",
            embedding_function=self.embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        self.ingestor = DocumentIngestor()

    def add_documents(self):
        """Processes all documents in data/ and adds them to Chroma."""
        # 1. Fetch existing URLs for Idempotency
        existing_urls = set()
        try:
            existing_data = self.vector_store.get(include=["metadatas"])
            if existing_data and existing_data.get("metadatas"):
                for meta in existing_data["metadatas"]:
                    if meta and "source_url" in meta:
                        existing_urls.add(meta["source_url"])
        except Exception as e:
            print(f"Warning: Could not fetch existing metadata: {e}")

        all_docs = []
        skipped_count = 0
        for file_path in glob.glob(os.path.join(DATA_DIR, "*")):
            # 2. Filter Logic: source_url defaults to the exact file_path in our pipeline
            if file_path in existing_urls:
                print(f"Skipping already ingested file: {os.path.basename(file_path)}")
                skipped_count += 1
                continue
                
            if file_path.endswith(".pdf"):
                docs = self.ingestor.load_pdf(file_path)
                all_docs.extend(docs)
            elif file_path.endswith((".txt", ".md")):
                docs = self.ingestor.load_markdown(file_path)
                all_docs.extend(docs)
                
        print(f"Skipped {skipped_count} previously ingested files.")
        
        if not all_docs:
            print("No new documents to ingest. Vector store is up to date.")
            return

        # 3. Process Remaining
        if all_docs:
            import time
            chunks = self.ingestor.process_and_chunk(all_docs)
            print(f"Prepared {len(chunks)} new chunks for ingestion.")
            
            batch_size = 15
            total_chunks = len(chunks)
            num_batches = (total_chunks + batch_size - 1) // batch_size
            
            print(f"Starting ingestion of {total_chunks} chunks in {num_batches} batches...")
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_chunks)
                batch_chunks = chunks[start_idx:end_idx]
                
                # Safety Net: Try-Exception Retry Loop
                success = False
                while not success:
                    try:
                        self.vector_store.add_documents(documents=batch_chunks)
                        success = True
                    except Exception as e:
                        print(f"Exception caught (potentially 429 Rate Limit): {e}")
                        print("Cooling down for 30 seconds before retrying batch...")
                        time.sleep(30)
                
                progress_percentage = (end_idx / total_chunks) * 100
                print(f"Progress: Chunk {end_idx} of {total_chunks} ({progress_percentage:.1f}%) complete.")
                
                if i < num_batches - 1:
                    time.sleep(12)
                    
            print(f"Successfully ingested all {total_chunks} chunks into vector store at {VECTORSTORE_DIR}.")

# Global Singleton to prevent SQLite Tenant file locks
GLOBAL_MANAGER = VectorStoreManager()

class PolicySearchTool(BaseTool):
    name: str = "policy_search_tool"
    description: str = (
        "Returns the top 2 most relevant policy excerpts. "
        "Always pass a specific search query like 'return window for electronics' or 'lost package procedure'."
    )

    def _run(self, query: str) -> str:
        # Fetch top 2 most relevant documents utilizing the Persistent Singleton
        retriever = GLOBAL_MANAGER.vector_store.as_retriever(search_kwargs={"k": 2})
        docs = retriever.invoke(query)
        
        formatted_results = []
        for doc in docs:
            citation = utils.format_citation(
                doc.metadata.get("source_url", ""), 
                doc.metadata.get("section_heading", "")
            )
            # Format explicitly linking citation to excerpt
            excerpt = f"CITATION: {citation}\nEXCERPT: {doc.page_content}"
            formatted_results.append(excerpt)
            
        if not formatted_results:
            return "No relevant policies found for this query."
            
        return "\n\n---\n\n".join(formatted_results)

if __name__ == "__main__":
    manager = VectorStoreManager()
    manager.add_documents()
