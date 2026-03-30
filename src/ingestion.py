"""
Data Ingestion Pipeline for the Agentic RAG Foundation.

Handles loading of raw Markdown and PDF documents, chunking text via
RecursiveCharacterTextSplitter, and mathematically ensuring metadata 
(source_url, section_heading) is correctly embedded in every chunk.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentIngestor:
    """
    Responsible for ingesting, chunking, and tagging documents for the vectorstore.
    Enforces the presence of citation-critical metadata elements.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initializes the ingestor with the required text splitting policy.
        
        Args:
            chunk_size (int): Expected size of each text block (Default 800).
            chunk_overlap (int): Amount of overlapping token/characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # RecursiveCharacterTextSplitter handles splitting on paragraphs, then sentences, 
        # then words, safely maintaining semantic structure as much as possible.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def load_pdf(self, file_path: str, source_url: Optional[str] = None) -> List[Document]:
        """
        Loads a PDF file, appending basic spatial metadata.
        
        Args:
            file_path (str): The local path to the PDF document.
            source_url (str, optional): The designated URL/Origin to tag.
            
        Returns:
            List[Document]: The loaded pages wrapped as LangChain Documents.
        """
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        target_url = source_url if source_url else file_path
        
        for i, doc in enumerate(docs):
            # Seed preliminary metadata structure
            doc.metadata["source_url"] = target_url
            doc.metadata["section_heading"] = f"Page {doc.metadata.get('page', i+1)}"
            
        return docs

    def load_markdown(self, file_path: str, source_url: Optional[str] = None) -> List[Document]:
        """
        Loads a raw Markdown or plain text document.
        
        Args:
            file_path (str): The local path to the Text document.
            source_url (str, optional): The designated URL/Origin to tag.
            
        Returns:
            List[Document]: The loaded file wrapped as a LangChain Document.
        """
        # Specify utf-8 to prevent common Windows decoding errors on symbols
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        
        target_url = source_url if source_url else file_path
        
        for doc in docs:
            # Seed preliminary metadata structure
            doc.metadata["source_url"] = target_url
            # Defaults to Global context; later chunking logic will append IDs
            doc.metadata["section_heading"] = "Global Text Content"
            
        return docs

    def process_and_chunk(self, documents: List[Document]) -> List[Document]:
        """
        Splits loaded source documents into standard overlapping chunks and 
        finalizes critical metadata vectors.
        
        Args:
            documents (List[Document]): The loaded, full-text Documents.
            
        Returns:
            List[Document]: Milled document chunks ready for embedding.
        """
        # LangChain text splitter preserves metadata from the parent Document 
        # while slicing the `page_content`.
        chunks = self.text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            # Failsafe: Ensure critical metadata fields exist
            if "source_url" not in chunk.metadata:
                chunk.metadata["source_url"] = "Unknown URL"
                
            # Guarantee a unique section_heading/chunk_id representation
            if "section_heading" not in chunk.metadata:
                chunk.metadata["section_heading"] = f"Chunk_{i}"
            else:
                original_heading = chunk.metadata["section_heading"]
                chunk.metadata["section_heading"] = f"{original_heading} | Chunk_{i}"
                
        return chunks
