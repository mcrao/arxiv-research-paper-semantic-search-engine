import numpy as np

def dense_search(collection, embedding_model, query, top_k=5):
    q_emb = embedding_model.encode([query])
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

    res = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=top_k
    )

    results = []
    for i in range(len(res["documents"][0])):
        results.append({
            "title": res["metadatas"][0][i]["title"],
            "abstract": res["documents"][0][i],
            "score_dense": 1 - res["distances"][0][i]
        })

    return results
