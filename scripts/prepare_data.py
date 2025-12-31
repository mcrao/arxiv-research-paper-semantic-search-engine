import pickle
from datasets import load_dataset

dataset = load_dataset("MaartenGr/arxiv_nlp")["train"]

titles = list(dataset["Titles"])
abstracts = [a.strip().replace("\n", " ") for a in dataset["Abstracts"]]

with open("data/arxiv_data.pkl", "wb") as f:
    pickle.dump(
        {"titles": titles, "abstracts": abstracts},
        f
    )
