import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPEN_API_KEY")

os.environ["HF_HOME"] = r"D:\huggingface_models"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\huggingface_models"

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

qa_prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions based only on the given context.
Provide the answer in a concise and clear manner.
If the answer is not found in the context, say "I don't know".

<context>
{context}
</context>

Question: {input}
""")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_folder=r"F:\huggingface_models",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def create_embeddings():
    if "vectorstore" not in st.session_state:
        with st.spinner("Creating vector embeddings..."):
            embeddings = get_embeddings()


            loader = PyPDFDirectoryLoader("research_papers")
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            final_documents = text_splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(final_documents, embeddings)

            st.session_state.embeddings = embeddings
            st.session_state.documents = documents
            st.session_state.final_documents = final_documents
            st.session_state.vectorstore = vectorstore

        st.success("Vector embeddings created successfully.")

st.title("Research Paper Chatbot")

user_prompt = st.text_input("Enter your question here")

if st.button("Create Vector Store"):
    create_embeddings()

if user_prompt:
    create_embeddings()

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=qa_prompt,
        output_parser=StrOutputParser()
    )

    retriever = st.session_state.vectorstore.as_retriever()

    retrieval_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    end_time = time.process_time()

    st.write("Response:")
    st.write(response["answer"])

    st.write(f"Processing Time: {end_time - start_time:.2f} seconds")

    with st.expander("Show Retrieved Documents"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------")