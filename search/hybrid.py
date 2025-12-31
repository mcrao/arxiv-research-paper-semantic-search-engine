def hybrid_rerank(query, dense_results, bm25_results, reranker, top_k=5):
    merged = {}

    for r in dense_results + bm25_results:
        merged[r["abstract"]] = r

    candidates = list(merged.values())
    pairs = [(query, c["abstract"]) for c in candidates]

    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["score_rerank"] = float(s)

    return sorted(
        candidates,
        key=lambda x: x["score_rerank"],
        reverse=True
    )[:top_k]
