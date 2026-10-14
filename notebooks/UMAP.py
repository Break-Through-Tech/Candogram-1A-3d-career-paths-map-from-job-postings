from pathlib import Path
import umap
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

NOTEBOOKS_DIR = Path(__file__).resolve().parent
ROOT_DIR = NOTEBOOKS_DIR.parent
DATA_DIR = ROOT_DIR / "data"

embeddings_path = DATA_DIR / "embeddings.npy"
csv_path = DATA_DIR / "embedding_input.csv"

embeddings = np.load(embeddings_path)
df = pd.read_csv(csv_path)

df['Business Title'] = df['Business Title'].fillna('N/A')
df['Career Level'] = df['Career Level'].fillna('Unknown').astype(str)

if len(embeddings) != len(df):
    raise ValueError(f"You halve {len(embeddings)} embeddings but {len(df)} rows in your CSV 😡.")

#squashes the embeddings to 50 dimensions for nearest neighbors
reducer_50d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=50, random_state=42)
embeddings_50d = reducer_50d.fit_transform(embeddings)

#squashes the embeddings to 2 dimensions for 3d visualization
reducer_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embeddings_2d = reducer_2d.fit_transform(embeddings_50d)

embeddings_50dim = np.asarray(embeddings_50d, dtype=np.float32)
embeddings_2dim = np.asarray(embeddings_2d, dtype=np.float32)

np.save(DATA_DIR / "embeddings_50d.npy", np.asarray(embeddings_50d, dtype=np.float32))
np.save(DATA_DIR / "embeddings_2d.npy", np.asarray(embeddings_2d, dtype=np.float32))

#this converts the career levels into numbers (e.g. Entry-Level = 0, Manager = 2)
categories = df['Career Level'].astype('category')
numeric_categories = categories.cat.codes.values.reshape(-1, 1)

scaler = MinMaxScaler()
z_scale = scaler.fit_transform(numeric_categories.astype(float)).flatten()

plot_df = pd.DataFrame({
    'UMAP_1': embeddings_2d[:, 0],
    'UMAP_2': embeddings_2d[:, 1],
    'Career_Level_Scaled': z_scale,
    'Career_Level': df['Career Level'],
    'Business_Title': df['Business Title']
})

fig = px.scatter_3d(
    plot_df,
    x='UMAP_1',
    y='UMAP_2',
    z='Career_Level_Scaled',
    color='Career_Level',
    hover_data=['Business_Title'],
    title="2D UMAP Embeddings with Career Level as Z-Axis"
)

fig.update_traces(marker=dict(size=3, opacity=0.8))
fig.update_layout(
    scene=dict(
        zaxis_title='Career Level '
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()