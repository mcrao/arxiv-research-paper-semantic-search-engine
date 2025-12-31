import pickle
from rank_bm25 import BM25Okapi

def load_raw_data():
    with open("data/arxiv_data.pkl", "rb") as f:
        data = pickle.load(f)

    titles = data["titles"]
    abstracts = data["abstracts"]

    tokenized = [a.lower().split() for a in abstracts]
    bm25 = BM25Okapi(tokenized)

    return titles, abstracts, bm25
