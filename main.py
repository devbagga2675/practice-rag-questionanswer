# simple_rag_qa.py
# --- Installation ---
# pip install streamlit langchain langchain-google-genai langchain-community langchain-chroma pypdf python-docx

import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import tempfile
import os
from dotenv import load_dotenv
import nest_asyncio # <-- 1. Import the library

nest_asyncio.apply()
# Load environment variables from .env file
load_dotenv()
# --- Streamlit UI Setup ---
st.set_page_config(page_title="Simple RAG Q&A", layout="wide")
st.title("💬 Simple RAG Q&A")
st.write(
    """
    This app demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline. 
    Upload a document (PDF or DOCX), and ask questions about its content.
    """
)

# --- API Key Management ---
try:
    # gemini_api_key = st.secrets["GOOGLE_API_KEY"]
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
except KeyError:
    st.error("GOOGLE_API_KEY is not set. Please add it to your Streamlit secrets.")
    st.stop()

# --- RAG Pipeline Logic ---
@st.cache_resource(show_spinner="Processing document...")
def process_document(uploaded_file):
    """Loads, splits, embeds, and stores a document in a vector store."""
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load the document
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            loader = Docx2txtLoader(tmp_file_path)
        else:
            st.error("Unsupported file type.")
            return None
        
        documents = loader.load()
        os.remove(tmp_file_path) # Clean up the temporary file

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings and store in ChromaDB
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
        vector_store = Chroma.from_documents(chunks, embeddings)
        
        return vector_store.as_retriever(search_kwargs={"k": 3})

    except Exception as e:
        st.error(f"An error occurred during document processing: {e}")
        return None

async def get_rag_response(retriever, query: str):
    """Generates a response using the RAG chain."""
    
    # 1. Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=gemini_api_key)
    
    # 2. Define the RAG prompt template
    prompt_template = ChatPromptTemplate.from_template("""
    You are an expert assistant. Answer the user's question based *only* on the following context.
    If the context does not contain the answer, state that you don't know.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """)

    # 3. Create the RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # 4. Invoke the chain asynchronously
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
        
        # Initialize chat history
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        # Display chat messages from history
        for message in st.session_state.rag_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Main chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = asyncio.run(get_rag_response(retriever, prompt))
                st.markdown(response)
            
            st.session_state.rag_messages.append({"role": "assistant", "content": response})