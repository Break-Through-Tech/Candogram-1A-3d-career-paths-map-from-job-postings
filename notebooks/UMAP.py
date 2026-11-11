from pathlib import Path
import umap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

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

# squashes the embeddings to 50 dimensions for nearest neighbors
reducer_50d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=50, random_state=42)
embeddings_50d = reducer_50d.fit_transform(embeddings)

# squashes the embeddings to 2 dimensions for 3d visualization
reducer_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embeddings_2d = reducer_2d.fit_transform(embeddings_50d)

np.save(DATA_DIR / "embeddings_50d.npy", np.asarray(embeddings_50d, dtype=np.float32))
np.save(DATA_DIR / "embeddings_2d.npy", np.asarray(embeddings_2d, dtype=np.float32))

level_map = {"Student": 1, "Entry-Level": 2, "Experienced (non-manager)": 3,
             "Manager": 4, "Executive": 5}
z_scale = df['Career Level'].map(level_map).to_numpy()

plot_df = pd.DataFrame({
    'row_id': np.arange(len(df)),          # position in the CSV; matches the source/target in edges.csv
    'UMAP_1': embeddings_2d[:, 0],
    'UMAP_2': embeddings_2d[:, 1],
    'Career_Level_Scaled': z_scale,
    'Career_Level': df['Career Level'],
    'Business_Title': df['Business Title']
})

edges = pd.read_csv(DATA_DIR / "edges.csv")


def build_base_figure():
    """The dots only, with no edges."""
    fig = px.scatter_3d(
        plot_df,
        x='UMAP_1',
        y='UMAP_2',
        z='Career_Level_Scaled',
        color='Career_Level',
        hover_data=['Business_Title'],
        custom_data=['row_id'],             # lets the callback know WHICH job was clicked
        title="2D UMAP Embeddings with Career Level as Z-Axis"
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(
        scene=dict(zaxis_title='Career Level '),
        margin=dict(l=0, r=0, b=0, t=40),
        uirevision="keep",                  # keeps the camera angle/zoom when the figure updates
        height=800,
    )
    return fig


def add_edges_for_job(fig, job_id):
    """Adds only the edges touching job_id (as source or target), plus a highlight on the connected dots."""
    touching = edges[(edges["source"] == job_id) | (edges["target"] == job_id)]

    line_x, line_y, line_z = [], [], []
    for s, t in zip(touching["source"], touching["target"]):
        line_x += [embeddings_2d[s, 0], embeddings_2d[t, 0], None]
        line_y += [embeddings_2d[s, 1], embeddings_2d[t, 1], None]
        line_z += [z_scale[s], z_scale[t], None]

    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z, mode="lines",
        line=dict(width=4, color="crimson"),
        hoverinfo="skip", showlegend=False,
    ))

    # outline the clicked dot and the jobs it connects to
    neighbor_ids = set(touching["source"]) | set(touching["target"]) | {job_id}
    ids = sorted(neighbor_ids)

    # job titles shown as always-visible labels (shortened so they don't pile up)
    titles = [
        t if len(t) <= 40 else t[:37] + "..."
        for t in df.loc[ids, "Business Title"].astype(str)
    ]

    fig.add_trace(go.Scatter3d(
        x=embeddings_2d[ids, 0], y=embeddings_2d[ids, 1], z=z_scale[ids],
        mode="markers+text",
        text=titles,
        textposition="top center",
        textfont=dict(size=11, color="black"),
        marker=dict(size=6, color="rgba(0,0,0,0)", line=dict(color="black", width=3)),
        hoverinfo="skip", showlegend=False,
    ))
    return fig


app = Dash(__name__)

app.layout = html.Div([
    html.Button("Clear edges", id="clear-btn", n_clicks=0),
    html.Span("  Click a dot to see the career paths connected to it.",
              style={"marginLeft": "10px"}),
    dcc.Graph(id="career-graph", figure=build_base_figure()),
    html.Div(id="selected-job", style={"padding": "8px", "fontWeight": "bold"}),
])


@app.callback(
    Output("career-graph", "figure"),
    Output("selected-job", "children"),
    Input("career-graph", "clickData"),
    Input("clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_edges(click_data, _n_clicks):
    from dash import ctx
    fig = build_base_figure()

    # "Clear edges" button was pressed -> dots only
    if ctx.triggered_id == "clear-btn":
        return fig, ""

    if not click_data:
        return fig, ""

    point = click_data["points"][0]
    custom = point.get("customdata")
    if not custom:
        return fig, ""

    job_id = int(custom[0])
    n_edges = int(((edges["source"] == job_id) | (edges["target"] == job_id)).sum())
    fig = add_edges_for_job(fig, job_id)

    label = (f"{df.loc[job_id, 'Business Title']} "
             f"({df.loc[job_id, 'Career Level']}) - {n_edges} connected paths")
    return fig, label


if __name__ == "__main__":
    app.run(debug=True)
