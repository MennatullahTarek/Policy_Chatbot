import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI

# ------------------- Load and Chunk Documents -------------------
@st.cache_data
def load_docs():
    docs = []
    for filename in os.listdir("docs"):
        loader = TextLoader(os.path.join("docs", filename), encoding="utf-8")
        docs.extend(loader.load())
    return docs

documents = load_docs()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

# ------------------- Embedding & Vector Store -------------------
@st.cache_resource
def create_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

vectordb = create_vectorstore()
retriever = vectordb.as_retriever()

# ------------------- LLM-based QA Chain (Gemini) -------------------
llm = VertexAI(model_name="gemini-1.5-flash", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Egypt Policy Assistant 🇪🇬")
st.title("🤖 Egypt Policy Chatbot (RAG + Gemini 1.5 Flash)")
st.markdown("Ask any question related to Egypt's national AI strategy or public governance.")

query = st.text_input("Ask your question:")

if query:
    with st.spinner("Searching the knowledge base..."):
        result = qa_chain.run(query)
    st.success("Answer:")
    st.write(result)

