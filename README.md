# 📄 Multi-PDF RAG Chatbot

A conversational AI application that allows users to upload multiple PDF documents and ask questions about their content.

The application uses **Retrieval-Augmented Generation (RAG)** with **LangChain, Google Gemini, and FAISS** to retrieve relevant information from the uploaded documents and generate accurate, document-grounded responses.

---

## 🚀 Project Overview

This project demonstrates how Large Language Models can be connected to external documents using RAG.

Users can upload multiple text-based PDF documents and ask questions in natural language. The application retrieves the most relevant information from the documents and provides it to Google Gemini as context before generating the final response.

---

## ✨ Features

- 📚 Upload multiple PDF documents
- 📖 Extract text from PDF files
- ✂️ Split documents into smaller text chunks
- 🧠 Generate embeddings using Google Gemini
- 🔍 Perform semantic similarity search using FAISS
- 🤖 Generate answers using Google Gemini
- 💬 Conversational question answering
- 📑 Display source PDF and page information
- 🌐 Interactive Streamlit interface
- 🔐 Secure API key management using environment variables

---

## 🧠 What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

RAG allows an LLM to use external information when generating an answer.

In this project:

```text
PDF Documents
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
Relevant Information Retrieval
      ↓
Google Gemini
      ↓
Generated Answer

Technologies Used
Python – Core programming language
LangChain – Document processing and RAG workflow
Google Gemini – Large Language Model and embeddings
FAISS – Vector similarity search
PyPDF – PDF text extraction
Streamlit – Web application interface
python-dotenv – Environment variable management

How It Works
1. Upload PDFs

The user uploads one or more text-based PDF documents through the Streamlit interface.

2. Extract Text

PyPDFLoader extracts the text from the uploaded PDF documents.

3. Split Text

RecursiveCharacterTextSplitter divides the extracted text into smaller chunks for efficient retrieval.

4. Generate Embeddings

Each text chunk is converted into a numerical vector using Google Gemini embeddings.

5. Store in FAISS

The generated embeddings are stored in a FAISS vector database.

6. Ask a Question

The user asks a natural-language question about the uploaded documents.

7. Retrieve Relevant Information

FAISS performs similarity search and retrieves the most relevant document chunks.

8. Generate Answer

The retrieved document context and the user's question are sent to Google Gemini.

Gemini generates the final answer based on the retrieved information.

9. Display Sources

The application displays the source PDF and page number associated with the retrieved information.
