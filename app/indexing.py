from langchain_community.document_loaders import PyMuPDFLoader
import os
def load_pdf(pdf_name):
    """Load pdf from uploads folder 

    Args:
        pdf_name (str): pdf name of file
    Returns:
        Document: a docs that contains 
    """
    # Construct the path to the file in the 'uploads' folder
    base_dir = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, '..', 'uploads', pdf_name)
    
    # Use PyMuPDFLoader to load the PDF
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # Here, you would add your logic for querying the loaded document (e.g., using RAG)
    # For now, just returning the loaded documents for testing
    return docs

