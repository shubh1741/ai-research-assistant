import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page title
st.title("AI Research Assistant")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector DB
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Local LLM
llm = ChatOllama(model="llama3")

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.

Context:
{context}

Question:
{question}
""")

# Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Display old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask something about your PDF")

if query:
    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    response = rag_chain.invoke(query)

    # Show assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })