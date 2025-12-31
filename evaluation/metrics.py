import numpy as np

def recall_at_k(relevant, retrieved, k):
    return len(set(relevant) & set(retrieved[:k])) / len(relevant)

def mrr(relevant, retrieved):
    for i, r in enumerate(retrieved):
        if r in relevant:
            return 1 / (i + 1)
    return 0

def ndcg(relevant, retrieved, k):
    dcg = 0
    for i, r in enumerate(retrieved[:k]):
        if r in relevant:
            dcg += 1 / np.log2(i + 2)

    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0
