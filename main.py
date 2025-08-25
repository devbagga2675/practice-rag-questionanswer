# simple_rag_qa_faiss_clean.py
# --- Installation ---
# pip install streamlit langchain langchain-google-genai langchain-community faiss-cpu pypdf python-docx python-dotenv nest-asyncio

import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import tempfile
import os
from dotenv import load_dotenv
import nest_asyncio

# Allows asyncio to run within Streamlit's existing event loop
nest_asyncio.apply()
load_dotenv()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Reliable RAG Q&A with FAISS", layout="wide")
st.title("💬 Reliable RAG Q&A with FAISS")
st.write(
    """
    This app uses the FAISS vector store to avoid common dependency issues.
    Upload a document (PDF or DOCX), and ask questions about its content.
    """
)

# --- API Key Management ---
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
    if not gemini_api_key:
        st.error("GOOGLE_API_KEY is not set. Please add it to your .env file or Streamlit secrets.")
        st.stop()
except (KeyError, FileNotFoundError):
    st.error("GOOGLE_API_KEY is not configured. Please create a .env file or set secrets in Streamlit Cloud.")
    st.stop()

# --- RAG Pipeline Logic ---
@st.cache_resource(show_spinner="Processing document...")
def process_document(uploaded_file):
    """Loads, splits, embeds, and stores a document in a FAISS vector store."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(tmp_file_path)
        else:
            st.error("Unsupported file type.")
            return None
        
        documents = loader.load()
        os.remove(tmp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.error(f"An error occurred during document processing: {e}")
        return None

async def get_rag_response(retriever, query: str):
    """Generates a response using the RAG chain."""
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=gemini_api_key)
    
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert assistant. Answer the user's question based *only* on the following context.
    If the context does not contain the answer, state that you don't know.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    with st.spinner("Searching document and generating answer..."):
        response = await chain.ainvoke(query)
    
    return response

# --- Streamlit UI for File Upload and Q&A ---
uploaded_file = st.file_uploader(
    "Upload your document (PDF or DOCX)", 
    type=["pdf", "docx"]
)

if uploaded_file:
    retriever = process_document(uploaded_file)
    if retriever:
        st.success("Document processed successfully! You can now ask questions.")
        
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = asyncio.run(get_rag_response(retriever, prompt))
                st.markdown(response)
            
            st.session_state.rag_messages.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a document to get started.")