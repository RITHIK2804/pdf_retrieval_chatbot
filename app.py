import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file.")
    st.stop()


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with Multiple PDFs")
st.write(
    "Upload multiple text-based PDF documents and ask questions about them."
)


# ============================================================
# 3. INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ============================================================
# 5. PDF UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload your PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# 6. PROCESS DOCUMENTS
# ============================================================

if st.button("🔄 Process Documents"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        st.stop()

    all_documents = []
    successfully_processed = []

    with st.spinner("Processing your PDF documents..."):

        for uploaded_file in uploaded_files:

            try:

                # ------------------------------------------------
                # Save uploaded PDF temporarily
                # ------------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name


                # ------------------------------------------------
                # Load PDF
                # ------------------------------------------------

                loader = PyPDFLoader(temp_file_path)

                documents = loader.load()


                # ------------------------------------------------
                # Add source information
                # ------------------------------------------------

                for document in documents:

                    document.metadata["source"] = uploaded_file.name

                    # PyPDFLoader pages are generally zero-indexed
                    if "page" in document.metadata:
                        document.metadata["page_number"] = (
                            document.metadata["page"] + 1
                        )

                all_documents.extend(documents)

                successfully_processed.append(
                    uploaded_file.name
                )


                # ------------------------------------------------
                # Remove temporary file
                # ------------------------------------------------

                os.remove(temp_file_path)


            except Exception as e:

                st.error(
                    f"Could not process {uploaded_file.name}: {e}"
                )


    # ========================================================
    # 7. CHECK WHETHER TEXT WAS EXTRACTED
    # ========================================================

    if not all_documents:

        st.error(
            "No readable text was extracted from the uploaded PDFs. "
            "Please use text-based PDFs."
        )

        st.stop()


    # ========================================================
    # 8. SPLIT DOCUMENTS INTO CHUNKS
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        all_documents
    )


    if not chunks:

        st.error(
            "No text chunks were created. "
            "Please upload text-based PDFs."
        )

        st.stop()


    # ========================================================
    # 9. CREATE VECTOR DATABASE
    # ========================================================

    with st.spinner("Creating embeddings and vector database..."):

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )


    # ========================================================
    # 10. STORE VECTOR DATABASE IN SESSION STATE
    # ========================================================

    st.session_state.vector_store = vector_store

    st.session_state.processed_files = successfully_processed

    # Clear previous chat when new documents are processed
    st.session_state.chat_history = []


    # ========================================================
    # 11. SHOW PROCESSING RESULTS
    # ========================================================

    st.success(
        f"✅ Processed {len(successfully_processed)} PDF(s), "
        f"{len(all_documents)} pages/documents, "
        f"and created {len(chunks)} text chunks."
    )


# ============================================================
# 12. DISPLAY PROCESSED FILES
# ============================================================

if st.session_state.processed_files:

    st.subheader("📄 Processed Documents")

    for filename in st.session_state.processed_files:
        st.write(f"✅ {filename}")


# ============================================================
# 13. DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# ============================================================
# 14. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your PDFs..."
)


# ============================================================
# 15. PROCESS QUESTION
# ============================================================

if question:

    # --------------------------------------------------------
    # Make sure documents have been processed
    # --------------------------------------------------------

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload and process your PDFs first."
        )

        st.stop()


    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # --------------------------------------------------------
    # Save user question
    # --------------------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )


    # ========================================================
    # 16. RETRIEVER
    # ========================================================

    retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={
            "k": 4
        }
    )


    # ========================================================
    # 17. RETRIEVE RELEVANT DOCUMENTS
    # ========================================================

    with st.spinner("Searching your documents..."):

        relevant_documents = retriever.invoke(
            question
        )


    # ========================================================
    # 18. BUILD CONTEXT
    # ========================================================

    context_parts = []

    for document in relevant_documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page_number",
            "Unknown"
        )

        context_parts.append(
            f"""
Source: {source}
Page: {page}

Content:
{document.page_content}
"""
        )


    context = "\n\n".join(
        context_parts
    )


    # ========================================================
    # 19. BUILD CHAT HISTORY
    # ========================================================

    previous_chat = ""

    for message in st.session_state.chat_history[-6:]:

        previous_chat += (
            f"{message['role'].upper()}: "
            f"{message['content']}\n"
        )


    # ========================================================
    # 20. RAG PROMPT
    # ========================================================

    prompt = f"""
You are a helpful document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided document context.

Rules:

1. Do not invent information.
2. If the answer is not present in the documents,
   say that the information is not available
   in the uploaded documents.
3. Use the previous conversation when necessary
   to understand follow-up questions.
4. Give a clear and concise answer.
5. When possible, mention the relevant PDF source
   and page number.

DOCUMENT CONTEXT
================

{context}


PREVIOUS CONVERSATION
=====================

{previous_chat}


USER QUESTION
=============

{question}
"""


    # ========================================================
    # 21. SEND TO GEMINI
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner("Generating answer..."):

            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                answer = "\n".join(
                block.get("text", "")
                for block in response.content
                if isinstance(block, dict)
            )
            else:
                answer = str(response.content)

            st.markdown(answer)

    # ========================================================
    # 22. SAVE ASSISTANT RESPONSE
    # ========================================================

st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # ========================================================
    # 23. SHOW RETRIEVED SOURCES
    # ========================================================

with st.expander("📚 Sources used for this answer"):

        for document in relevant_documents:

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page_number",
                "Unknown"
            )

            st.write(
                f"📄 {source} — Page {page}"
            )