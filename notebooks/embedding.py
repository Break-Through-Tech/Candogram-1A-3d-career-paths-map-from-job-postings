#I chose a huggingface embedding model.
#You must set up an account along with a couple quick changes to get started

import pandas as pd
import os
import numpy as np
import json
from sentence_transformers import SentenceTransformer

file_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(file_dir, "..", "data", "embedding_input.csv")
output_path = os.path.join(file_dir, "..", "data", "embeddings.npy")
json_path = os.path.join(file_dir, "..", "data", "embedding_paths.json")

df = pd.read_csv(file_path)

sentences = df['cleaned descriptions'].astype(str).tolist()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences, show_progress_bar=True, batch_size = 16)

#//////////////////////////////////////////////////////////////////////////////#

#Change Name
np.save(output_path, embeddings)

#with open(f"{file_dir}/".." / "data" / "data _paths.json", "w") as f:
 #   json.dump([str(path) for path in txt_files], f)

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(sentences, f, indent=4)

print("The embeddings have been saved 😻😻😻")

