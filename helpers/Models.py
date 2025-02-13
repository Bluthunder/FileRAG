from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint


def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        max_new_tokens=512,
        top_k=30,
        temperature=0.1,
        repetition_penalty=1.03,
    )
    # llm = HuggingFaceEmbeddings(repo_id=repo_id, task=task, max_new_token=, top_k=top_k, temperature=temp,
    #                             repetation_penalty=rep_penalty)
    return llm


def get_embedding(model_path):
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(model_name=model_path, encode_kwargs=encode_kwargs)
    print(f' Embeding {embedding}')
    return embedding
