"""
Utility functions for the Agentic RAG Foundation.

Provides helper mechanisms such as consistent citation formatting 
which is strictly mandated by the system guidelines.
"""

def format_citation(source_url: str, section_heading: str) -> str:
    """
    Formats a citation string to strictly match the mandated format: 
    [Source URL + Section/Chunk ID].
    
    Args:
        source_url (str): The canonical URL, hyperlink, or origin policy document name.
        section_heading (str): The specific chunk ID, section, or page number.
        
    Returns:
        str: A formatted citation string ready to be embedded by the Resolution Writer.
    """
    # Sanitize inputs to prevent malformed formatting
    safeguard_url = source_url if source_url else "Unknown Source"
    safeguard_heading = section_heading if section_heading else "Unknown Section"
        
    return f"[{safeguard_url} + {safeguard_heading}]"
