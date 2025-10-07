import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
import tempfile
import os

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="centered")
st.title("📚 Chat with your PDF (LangChain + Chroma)")
st.write("Upload a PDF and ask questions about its content using a lightweight open LLM.")

# 🔹 Set Hugging Face token (add in Streamlit Cloud secrets)
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    st.warning("⚠️ Please set your Hugging Face API token in Streamlit Cloud Secrets.")
else:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ["HUGGINGFACEHUB_API_TOKEN"]

# File uploader
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info("✅ PDF uploaded successfully!")

    # Load and split
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings and vector store
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Load a small, fast model from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0, "max_length": 256}
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Chat interface
    query = st.text_input("💬 Ask a question about your PDF:")

    if st.button("Get Answer") and query:
        with st.spinner("Thinking..."):
            response = qa.run(query)
        st.success("✅ Answer:")
        st.write(response)
