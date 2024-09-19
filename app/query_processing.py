from .retriever import generate_hypothetical_docs, bm25_retriever, ensemble_retriever, reciprocal_rank_fusion, store_docs, parent_child_retriever
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from .indexing import load_pdf
from operator import itemgetter

def query_pdf(pdf_name, question):
    """
    Processes a PDF document to answer a given question using a combination of retrievers and a language model.

    Args:
        pdf_name (str): The name of the PDF file to be processed.
        question (str): The question to be answered based on the content of the PDF.

    Returns:
        str: The generated answer to the question based on the context from the PDF.
    """
    prompt_template="""
    Use the following piece of context to answer the question asked.
    Please try to provide the answer only based on the context
    {context}
    Question:{question}
    Helpful Answers:
    """
    prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])
    
    docs = load_pdf(pdf_name)
    vector_store = store_docs()
    pc_retriever = parent_child_retriever(vector_store, docs)
    bm_retriever = bm25_retriever(docs)
    retriever = ensemble_retriever(pc_retriever, bm_retriever)
    
    hyDE_chain = generate_hypothetical_docs(model_name="gpt-3.5-turbo")
    retriever_chain = hyDE_chain | retriever.map() | reciprocal_rank_fusion 
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    final_rag_chain = (
        {"context": retriever_chain,
        "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = final_rag_chain.invoke({"question": question})
    
    return response
