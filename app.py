from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

from helpers.Models import get_embedding, get_llm
from helpers.Prompt import prompt_template
from helpers.Upload import upload_and_save
from helpers.VectorStore import create_vector_db
from tools.DocumentProcessing import pdf_loader, document_split, directory_loader

import gradio as gr

FILE_PATH = 'files/Machine_Learning_Theory_QA_-AI___MLOps.pdf'
# FILE_PATH_1 = 'files/Star Wars - Jedi Academy Training Manual.pdf'
EMBEDDING_MODEL = 'mixedbread-ai/mxbai-embed-large-v1'
LLM_MODEL = 'HuggingFaceH4/zephyr-7b-beta'

uploaded_file = 'files/uploads/'


def get_context_info(question):
    # document = pdf_loader(uploaded_file)
    document = directory_loader(uploaded_file)
    split = document_split(document)
    embedding_model = get_embedding(model_path=EMBEDDING_MODEL)
    vector_db = create_vector_db(split, embedding_model)
    retriever = vector_db.as_retriever()
    doc = retriever.invoke(question)
    return doc


def query_rag(question_input):
    llm = get_llm()

    retrieval = RunnableParallel(
        {
            "context": RunnableLambda(lambda x: get_context_info(x["question"])),

            "question": RunnableLambda(lambda x: x["question"])
        }
    )

    rag_chain = (retrieval |
                 prompt_template() |
                 llm |
                 StrOutputParser())

    response = rag_chain.invoke({'question': question_input})
    print(f'Response ---> {response}')
    return response


if __name__ == '__main__':
    load_dotenv()
    print("Hello, File Processing Started")

    with gr.Blocks() as app:
        gr.Markdown("## RAG-based QA from Documents")

        upload_button = gr.UploadButton("Click to Upload a file")
        upload_button.upload(upload_and_save, upload_button)
        question_input = gr.Textbox(label="Ask a Question")
        answer_output = gr.Textbox(label="Answer", interactive=False)
        query_button = gr.Button("Get Answer")
        query_button.click(query_rag, inputs=[question_input], outputs=[answer_output])

    app.launch()
