from langchain_core.prompts import PromptTemplate


def prompt_template():
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, that is if the answer is not in the context, then just say that you don't know, don't try to make up an answer.
    Always say "thanks for asking!" at the end of the answer.

    {context}
    Question: {question}
    Helpful Answer:"""

    qa_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    return qa_prompt


