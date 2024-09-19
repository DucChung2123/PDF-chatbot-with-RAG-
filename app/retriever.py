from langchain.embeddings import HuggingFaceBgeEmbeddings
import torch 
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.load import dumps, loads
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def store_docs(embedding_model_name="BAAI/bge-small-en-v1.5"):
    """
    Stores documents in a vector store using embeddings from a specified model.

    Args:
        embedding_model_name (str, optional): The name of the embedding model to use. 
                                              Defaults to "BAAI/bge-small-en-v1.5".

    Returns:
        Chroma: A Chroma vector store containing the embedded documents.
    """
    model_name = embedding_model_name
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs=encode_kwargs
    )
    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=bge_embeddings  #OpenAIEmbeddings()
    )
    return vectorstore
    
def parent_child_retriever(vectorstore, docs):
    """
    Creates a parent-child document retriever using the provided vector store and documents.

    Args:
        vectorstore: The vector store containing the embedded documents.
        docs: The documents to be added to the retriever.

    Returns:
        ParentDocumentRetriever: An instance of ParentDocumentRetriever configured with the provided vector store and documents.
    """
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    store = InMemoryStore()

    parent_child_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    parent_child_retriever.add_documents(docs)
    
    return parent_child_retriever
def bm25_retriever(docs):
    """
    Creates a BM25 retriever from the provided documents.

    Args:
        docs: The documents to be indexed by the BM25 retriever.

    Returns:
        BM25Retriever: An instance of BM25Retriever configured with the provided documents.
    """
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 2
    return bm25_retriever

def ensemble_retriever(dense_retriever, sparse_retriever):
    """
    Creates an ensemble retriever that combines dense and sparse retrievers.

    Args:
        dense_retriever: The dense retriever to be included in the ensemble.
        sparse_retriever: The sparse retriever to be included in the ensemble.

    Returns:
        EnsembleRetriever: An instance of EnsembleRetriever configured with the provided dense and sparse retrievers.
    """
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

def reciprocal_rank_fusion(results: list[list], k=60):
    """
    Performs Reciprocal Rank Fusion (RRF) on a list of ranked document lists.

    Args:
        results (list[list]): A list of lists, where each inner list contains documents ranked by relevance.
        k (int, optional): A constant used in the RRF formula to dampen the impact of rank. Defaults to 60.

    Returns:
        list[tuple]: A list of tuples where each tuple contains a document and its fused score, sorted by the fused score in descending order.
    """
    
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def generate_hypothetical_docs(model_name="gpt-3.5-turbo"):
    """
    Generates a chain for creating hypothetical scientific paper passages using a language model.

    Args:
        model_name (str, optional): The name of the language model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        Callable: A chain that generates hypothetical scientific paper passages in response to a given question.
    """
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    promt_template = """
    Please write 4 scientific paper passages to answer the question
    Question: {question}
    Passage:
    """
    promt = PromptTemplate.from_template(promt_template)

    generate_hypothetical_docs_chain = (
        promt 
        | llm 
        | StrOutputParser() 
        | (lambda x: x.split("\n"))
        | (lambda x: [i for i in x if i])
    )
    
    return generate_hypothetical_docs_chain