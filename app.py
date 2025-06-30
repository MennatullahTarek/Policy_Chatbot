# SQLite3 fix for Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_google_vertexai import VertexAI
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(
    page_title="Egypt Policy Assistant 🇪🇬",
    page_icon="🤖",
    layout="wide"
)

# ------------------- Initialize Session State -------------------
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# ------------------- Load and Chunk Documents -------------------
@st.cache_data
def load_docs():
    """Load documents from the docs directory."""
    docs = []
    docs_dir = "docs"
    
    if not os.path.exists(docs_dir):
        st.error(f"Documents directory '{docs_dir}' not found. Please create it and add your documents.")
        return []
    
    files = os.listdir(docs_dir)
    if not files:
        st.warning(f"No files found in '{docs_dir}' directory.")
        return []
    
    for filename in files:
        if filename.endswith('.txt'):
            try:
                loader = TextLoader(os.path.join(docs_dir, filename), encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                st.warning(f"Could not load {filename}: {str(e)}")
    
    return docs

# ------------------- Create Vector Store -------------------
@st.cache_resource
def create_vectorstore(_documents):
    """Create vector store from documents."""
    if not _documents:
        return None
    
    try:
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(_documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create vector store with a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            vectordb = Chroma.from_documents(
                chunks, 
                embeddings,
                persist_directory=None  # Use in-memory storage
            )
        
        return vectordb
    
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# ------------------- Initialize LLM -------------------
@st.cache_resource
def initialize_llm():
    """Initialize the Vertex AI LLM."""
    try:
        # Check if Google Cloud credentials are available
        if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and not os.environ.get('GOOGLE_CLOUD_PROJECT'):
            st.error("Google Cloud credentials not found. Please set up authentication.")
            return None
        
        llm = VertexAI(
            model_name="gemini-1.5-flash",
            temperature=0.2,
            max_output_tokens=1024
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# ------------------- Main App -------------------
def main():
    st.title("🤖 Egypt Policy Chatbot")
    st.markdown("*RAG + Gemini 1.5 Flash + Chroma*")
    st.markdown("Ask any question related to Egypt's national AI strategy or public governance.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        st.info("Make sure you have:")
        st.markdown("1. A `docs/` folder with `.txt` files")
        st.markdown("2. Google Cloud credentials configured")
        
        if st.button("🔄 Reload Documents"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.vectordb = None
            st.session_state.qa_chain = None
            st.rerun()
    
    # Load documents
    with st.spinner("Loading documents..."):
        documents = load_docs()
    
    if not documents:
        st.stop()
    
    st.success(f"✅ Loaded {len(documents)} documents")
    
    # Create vector store
    if st.session_state.vectordb is None:
        with st.spinner("Creating vector store..."):
            st.session_state.vectordb = create_vectorstore(documents)
    
    if st.session_state.vectordb is None:
        st.error("Failed to create vector store")
        st.stop()
    
    # Initialize LLM and QA chain
    if st.session_state.qa_chain is None:
        with st.spinner("Initializing AI model..."):
            llm = initialize_llm()
            if llm is None:
                st.stop()
            
            retriever = st.session_state.vectordb.as_retriever(
                search_kwargs={"k": 3}
            )
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
    
    st.success("✅ System ready!")
    
    # Query interface
    st.markdown("---")
    query = st.text_input("💬 Ask your question:", placeholder="What is Egypt's AI strategy?")
    
    if query:
        with st.spinner("🔍 Searching the knowledge base..."):
            try:
                result = st.session_state.qa_chain({"query": query})
                
                st.markdown("### 🤖 Answer:")
                st.write(result['result'])
                
                # Show source documents
                if 'source_documents' in result and result['source_documents']:
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(result['source_documents']):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(doc.page_content[:300] + "...")
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.json(doc.metadata)
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

if __name__ == "__main__":
    main()
