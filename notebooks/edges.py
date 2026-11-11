from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# generates file path, current file --> projects folder --> data folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data" 

# load the data
embeddings = np.load(DATA_DIR / "embeddings_50d.npy")

df = pd.read_csv(DATA_DIR / "embedding_input.csv")

# career levels into numbers
level_map = {"Student" : 1, "Entry-Level" : 2, "Experienced (non-manager)" : 3, "Manager" : 4, "Executive" : 5}

# replaces each name with career level number, a missing level is NaN
levels = df["Career Level"].map(level_map).to_numpy()

edges = []

for level in range(1, 5):
    lower = np.where(levels == level)[0]
    upper = np.where(levels == level+1)[0]

    knn = NearestNeighbors(n_neighbors=3)
    knn.fit(embeddings[upper])
    distances, neighbors = knn.kneighbors(embeddings[lower])

    for i, job in enumerate(lower):
        for d, n in zip(distances[i], neighbors[i]):
            edges.append((job, upper[n], d))

edges_df = pd.DataFrame(edges, columns=["source", "target", "distance"])
edges_df.to_csv(DATA_DIR / "edges.csv", index=False)
print(f"Saved {len(edges_df)} edges")

