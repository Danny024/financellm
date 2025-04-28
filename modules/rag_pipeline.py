import faiss
import os
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
import numpy as np

def setup_vector_store(chunks, base_url="http://localhost:11434"):
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url=base_url)
    single_vector = embeddings.embed_query("this is some text data")
    index = faiss.IndexFlatL2(len(single_vector))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def rerank_documents(question, docs, top_k=5):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[question, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [docs[i] for i in ranked_indices]

def create_rag_chain(retriever, base_url="http://localhost:11434", use_openai=False):
    prompt_template = ChatPromptTemplate.from_template("""
        You are an assistant for question-answering tasks. Use the following pieces of retrieved content to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from context.
        Question: {question}
        Context: {context}
        Answer:
    """)
    if use_openai and os.getenv("OPENAI_API_KEY"):
        model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    else:
        model = ChatOllama(model="deepseek-r1:1.5b", base_url=base_url)
    
    def rerank_and_format_docs(docs, question):
        reranked_docs = rerank_documents(question, docs, top_k=len(docs))
        return format_docs(reranked_docs)
    
    chain = (
        {"context": lambda x: rerank_and_format_docs(retriever.invoke(x), x), "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return chain