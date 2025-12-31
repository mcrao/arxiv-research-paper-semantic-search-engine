import numpy as np

def bm25_search(bm25, titles, abstracts, query, top_k=5):
    scores = bm25.get_scores(query.lower().split())
    idxs = np.argsort(scores)[-top_k:][::-1]

    return [{
        "title": titles[i],
        "abstract": abstracts[i],
        "score_bm25": float(scores[i])
    } for i in idxs]
