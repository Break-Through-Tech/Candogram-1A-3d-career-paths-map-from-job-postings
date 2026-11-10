import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap

root = Path(__file__).resolve().parent
data_dir = root / 'data'
embeddings_path = data_dir / 'embeddings.npy'
input_path = data_dir / 'embedding_input.csv'
output_path = data_dir / 'umap_points.json'

if not embeddings_path.exists():
    raise FileNotFoundError(f'Missing embeddings file: {embeddings_path}')

if not input_path.exists():
    raise FileNotFoundError(f'Missing embedding input file: {input_path}')

embeddings = np.load(embeddings_path)
df = pd.read_csv(input_path)

if len(df) != len(embeddings):
    raise ValueError(
        f'Embedding row count mismatch: {len(df)} CSV rows vs {len(embeddings)} embedding vectors.'
    )

reducer = umap.UMAP(
    n_components=3,
    n_neighbors=20,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
)
points = reducer.fit_transform(embeddings)

records = []
for index, row in df.iterrows():
    point = points[index]
    records.append({
        'jobId': int(row.get('Job ID', index)),
        'businessTitle': str(row.get('Business Title', '')).strip() or 'Unknown title',
        'agency': str(row.get('Agency', '')).strip() or 'Unknown agency',
        'careerLevel': str(row.get('Career Level', 'Unknown')).strip() or 'Unknown',
        'x': float(point[0]),
        'y': float(point[1]),
        'z': float(point[2]),
    })

output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Generated {len(records)} UMAP points at {output_path}')
