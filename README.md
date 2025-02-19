# 🌟 **File RAG - QA on Uploaded Files**

## 📝 Overview
File RAG is a Retrieval-Augmented Generation (RAG) application that enables users to upload files and perform question-answering (QA) using state-of-the-art language models. It leverages:

- 🔗 **LangChain** for orchestration and retrieval mechanisms.
- 🧠 **Mixedbread-ai/mxbai-embed-large-v1** for embedding generation.
- 🤖 **HuggingFaceH4/zephyr-7b-beta** as the language model for answering queries.
- 🎨 **Gradio** for an intuitive user interface.

## 🚀 Features
- 📂 Upload various file types (e.g., PDFs, TXT, CSV).
- 🔍 Extract relevant content using LangChain document loaders.
- 🏗️ Generate embeddings with `mxbai-embed-large-v1`.
- 🎯 Retrieve relevant chunks using similarity search.
- 💬 Answer user queries using `zephyr-7b-beta`.
- 🖥️ Interactive Gradio UI for seamless usage.

## ⚙️ Installation

### 🔧 Prerequisites
Ensure you have Python installed (>=3.8) and set up a virtual environment using Pipenv:

```bash
pip install pipenv
pipenv install
pipenv shell
```

## ▶️ Usage

Run the Gradio application:

```bash
python app.py
```

This will launch a local Gradio interface where users can upload files and interact with the QA system.

## 🔄 File Processing Pipeline
1. 📤 **Upload File**: Users upload a document.
2. ✂️ **Chunking & Embedding**: The file is split into meaningful chunks, and embeddings are generated using `mxbai-embed-large-v1`.
3. 🧐 **Vector Search**: The query is embedded and matched with the most relevant chunks.
4. 🤖 **Response Generation**: `zephyr-7b-beta` processes the retrieved context and generates an answer.

## 📌 Roadmap
- 📑 Support for more file formats.
- 🔬 Enhancements in retrieval for better accuracy.
.

## 👤 Author
**bluthunder**

## 📜 License
This project is licensed under the MIT License.

