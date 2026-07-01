import streamlit as st
from dotenv import load_dotenv
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if "messages" not in st.session_state:
    st.session_state.messages = []

history = ""

for msg in st.session_state.messages[-6:]:
    history += f"{msg['role']}: {msg['content']}\n"
st.markdown("""
<style>

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stMarkdownContainer"]{
    background:#2d3748;
    padding:12px 16px;
    border-radius:16px;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
[data-testid="stMarkdownContainer"]{
    background:#1f2937;
    padding:12px 16px;
    border-radius:16px;
}

/* User avatar right side */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){
    flex-direction:row-reverse;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div:nth-child(2){
    align-items:flex-end;
}

</style>
""", unsafe_allow_html=True)
# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="RAG Book Assistant",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<h1 style='text-align:center;'>📚 RAG Book Assistant</h1>",
    unsafe_allow_html=True
)

st.caption("Upload a PDF and chat with your document")

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf"
    )

    if uploaded_file:

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        st.success("PDF Uploaded")

        if st.button("Create Vector Database"):

            with st.spinner("Processing PDF..."):

                loader = PyPDFLoader(file_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(docs)

                embeddings = MistralAIEmbeddings()

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="chroma_db"
                )

                vectorstore.persist()

            st.success("Database Created!")

# -----------------------------
# LOAD VECTOR DB
# -----------------------------
if os.path.exists("chroma_db"):

    embeddings = MistralAIEmbeddings()

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(
        model="mistral-small-2506"
    )

    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Use the conversation history and the retrieved context
to answer the user's question.

Conversation History:
{history}

Context:
{context}

If the answer is not present in the context,
say:
"I could not find the answer in the document."
"""
        ),
        (
            "human",
            "{question}"
        )
    ]
)

    # -----------------------------
    # CHAT HISTORY
    # -----------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display old messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # -----------------------------
    # CHAT INPUT
    # -----------------------------
    user_question = st.chat_input(
        "Ask anything about your PDF..."
    )

    if user_question:

        # Show User Message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        # Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                docs = retriever.invoke(user_question)

                context = "\n\n".join(
                    [doc.page_content for doc in docs]
                )

                final_prompt = prompt.invoke(
                    {
                        "history": history,
                        "context": context,
                        "question": user_question
                    }
                )

                response = llm.invoke(final_prompt)

                answer = response.content

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

else:
    st.info(
        "Upload a PDF and create the vector database first."
    )

    
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
