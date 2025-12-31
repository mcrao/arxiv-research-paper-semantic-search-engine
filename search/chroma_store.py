import chromadb

def load_chroma_collection():
    client = chromadb.PersistentClient(
        path="./chroma_arxiv"
    )

    collection = client.get_collection("arxiv_abstracts")
    return collection
