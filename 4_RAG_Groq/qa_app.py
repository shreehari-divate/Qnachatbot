import os
import time
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())

st.set_page_config(page_title="Research Paper Chatbot")
st.title("Research Paper Chatbot")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.2
    )

@st.cache_resource
def get_embeddings():
    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        threads=1
    )

# No @st.cache_resource here — store result in session_state instead
def create_vectorstore():
    if not os.path.isdir("research_papers"):
        raise FileNotFoundError("'research_papers' folder not found. Create it and add PDFs inside.")
    
    loader = PyPDFDirectoryLoader("research_papers")
    documents = loader.load()
    
    if not documents:
        raise ValueError("No PDF files found inside the research_papers folder.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    final_documents = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    return vectorstore

qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based only on the given context.
If the answer is available in the context, answer clearly and concisely.
If the answer is not available in the context, say: I don't know.

Context:
{context}

Question:
{question}
""")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if st.button("Create Vector Store"):
    try:
        with st.spinner("Creating vector store..."):
            st.session_state.vectorstore = create_vectorstore()
        st.success("Vector store created successfully!")
    except Exception as e:
        st.error("Failed to create vector store.")
        st.exception(e)

# Use a form so it only fires on submit, not every keystroke
with st.form("qa_form"):
    user_prompt = st.text_input("Enter your question here")
    submitted = st.form_submit_button("Ask")

if submitted and user_prompt:
    if st.session_state.vectorstore is None:
        st.warning("Please click 'Create Vector Store' first.")
        st.stop()

    try:
        with st.spinner("Generating answer..."):
            start_time = time.time()

            docs = st.session_state.vectorstore.similarity_search(user_prompt, k=5)
            context = "\n\n".join(doc.page_content for doc in docs)

            llm = get_llm()
            chain = qa_prompt | llm | StrOutputParser()
            answer = chain.invoke({"context": context, "question": user_prompt})

            end_time = time.time()

        st.subheader("Answer")
        st.write(answer)
        st.caption(f"Processing time: {end_time - start_time:.2f} seconds")

        with st.expander("Show Retrieved Documents"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Document {i}**")
                st.write(doc.page_content)
                st.write("---")

    except Exception as e:
        st.error("Error while answering the question.")
        st.exception(e)