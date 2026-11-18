from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update

# ----------------------------------------------------------------------
# Paths + settings
# ----------------------------------------------------------------------
NOTEBOOKS_DIR = Path(__file__).resolve().parent
DATA_DIR = NOTEBOOKS_DIR.parent / "data"

embeddings_path = DATA_DIR / "embeddings.npy"
csv_path = DATA_DIR / "embedding_input.csv"
edges_path = DATA_DIR / "edges.csv"

# UMAP takes a while, so the 2D/50D results are cached on disk after the first run.
# Set to True only when you re-embed the jobs and want fresh coordinates.
REBUILD_UMAP = False

LEVELS = ["Student", "Entry-Level", "Experienced (non-manager)", "Manager", "Executive"]
LEVEL_NUM = {name: i + 1 for i, name in enumerate(LEVELS)}
LEVEL_SHORT = {
    "Student": "Student",
    "Entry-Level": "Entry-level",
    "Experienced (non-manager)": "Experienced",
    "Manager": "Manager",
    "Executive": "Executive",
}
LEVEL_COLOR = {
    "Student": "#17becf",
    "Entry-Level": "#2ca02c",
    "Experienced (non-manager)": "#ff9f1c",
    "Manager": "#7b4fd6",
    "Executive": "#d62728",
}

# camera presets for the "View" control (Plotly 3D already supports drag-rotate,
# scroll-zoom and toolbar pan; these give one-click viewpoints on top of that)
CAMERAS = {
    "3d": dict(eye=dict(x=1.6, y=1.6, z=0.9)),
    "side": dict(eye=dict(x=0.01, y=-2.6, z=0.35)),   # levels stacked like a ladder
    "top": dict(eye=dict(x=0.0, y=0.01, z=2.8)),      # pure similarity map
}

# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
embeddings = np.load(embeddings_path)
df = pd.read_csv(csv_path)

df["Business Title"] = df["Business Title"].fillna("N/A").astype(str)
df["Career Level"] = df["Career Level"].fillna("Unknown").astype(str)

if len(embeddings) != len(df):
    raise ValueError(f"You have {len(embeddings)} embeddings but {len(df)} rows in your CSV.")


def load_coordinates(emb):
    """Loads cached UMAP coordinates, or computes (and caches) them if missing."""
    p50 = DATA_DIR / "embeddings_50d.npy"
    p2 = DATA_DIR / "embeddings_2d.npy"

    if not REBUILD_UMAP and p2.exists():
        cached = np.load(p2)
        if len(cached) == len(emb):
            return cached

    import umap  # imported here so a normal launch doesn't pay the import cost

    # squashes the embeddings to 50 dimensions for nearest neighbors (used by edges.py)
    reducer_50d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=50, random_state=42)
    emb_50d = reducer_50d.fit_transform(emb)
    np.save(p50, np.asarray(emb_50d, dtype=np.float32))

    # squashes those to 2 dimensions for the map
    reducer_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    emb_2d = reducer_2d.fit_transform(emb_50d)
    np.save(p2, np.asarray(emb_2d, dtype=np.float32))
    return emb_2d


embeddings_2d = load_coordinates(embeddings)

X = embeddings_2d[:, 0]
Y = embeddings_2d[:, 1]
Z = df["Career Level"].map(LEVEL_NUM).to_numpy(dtype=float)  # NaN for "Unknown"
has_level = ~np.isnan(Z)  # jobs without a career level can't be placed on the z-axis
TITLES = df["Business Title"].to_numpy()
LEVEL_TXT = df["Career Level"].to_numpy()
DESCRIPTIONS = (
    df["cleaned descriptions"].fillna("").astype(str).to_numpy()
    if "cleaned descriptions" in df.columns
    else np.array([""] * len(df))
)

if edges_path.exists():
    edges = pd.read_csv(edges_path)
else:
    print("WARNING: edges.csv not found - run edges.py. The map will show no paths.")
    edges = pd.DataFrame(columns=["source", "target", "distance"])

# closest connections first, grouped once so clicking is instant
edges = edges.sort_values("distance")
UP_EDGES = {int(k): g["target"].astype(int).tolist() for k, g in edges.groupby("source")}
DOWN_EDGES = {int(k): g["source"].astype(int).tolist() for k, g in edges.groupby("target")}


def get_neighbors(job, direction, k):
    """Returns (jobs one level up, jobs one level down), closest first, at most k each."""
    ups = UP_EDGES.get(job, [])[:k] if direction in ("both", "up") else []
    downs = DOWN_EDGES.get(job, [])[:k] if direction in ("both", "down") else []
    return ups, downs


def shorten(text, n=30):
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------
HOVER = "<b>%{text}</b><br>%{customdata[1]}<extra></extra>"


def dot_trace(ids, level, size, opacity=0.85, outline=0.0, name=None):
    ids = np.asarray(ids, dtype=int)
    return go.Scatter3d(
        x=X[ids], y=Y[ids], z=Z[ids],
        mode="markers",
        name=name or LEVEL_SHORT[level],
        text=TITLES[ids],
        customdata=[[int(i), LEVEL_TXT[i]] for i in ids],  # row id, used by the click callback
        hovertemplate=HOVER,
        marker=dict(size=size, color=LEVEL_COLOR[level], opacity=opacity,
                    line=dict(color="white", width=outline)),
    )


def edge_trace(pairs, name, color, dash):
    xs, ys, zs = [], [], []
    for s, t in pairs:
        xs += [X[s], X[t], None]
        ys += [Y[s], Y[t], None]
        zs += [Z[s], Z[t], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", name=name,
        line=dict(width=7, color=color, dash=dash), hoverinfo="skip",
    )


def label_trace(selected, others):
    """
    Labels only for the selected job + its connections. Positions alternate around each
    dot (above / below / right / left) so neighbours that sit close together don't stack
    their text on top of each other. The side panel always has the full, readable list.
    """
    cycle = ["top center", "bottom center", "middle right", "middle left"]
    counter = {}
    ids, positions = [], []
    for j in sorted(others, key=lambda j: (Z[j], X[j])):
        n = counter.get(Z[j], 0)
        counter[Z[j]] = n + 1
        ids.append(j)
        positions.append(cycle[n % len(cycle)])
    return go.Scatter3d(
        x=X[ids], y=Y[ids], z=Z[ids], mode="text",
        text=[shorten(TITLES[j]) for j in ids], textposition=positions,
        textfont=dict(size=12, color="#222"), hoverinfo="skip", showlegend=False,
    )


def build_figure(selected, level_filter, direction, k, dot_size, options, view):
    fig = go.Figure()
    focus = selected is not None
    x_range = y_range = None

    if not focus:
        # BROWSE MODE: all dots (respecting the level filter), no edges
        for level in LEVELS:
            if level not in level_filter:
                continue
            ids = np.where(Z == LEVEL_NUM[level])[0]
            if len(ids):
                fig.add_trace(dot_trace(ids, level, size=dot_size, opacity=0.8))
    else:
        # FOCUS MODE: only the selected job and its connected jobs
        ups, downs = get_neighbors(selected, direction, k)
        path_ids = [selected] + ups + downs

        if "ghost" in options:
            rest = np.setdiff1d(np.where(has_level)[0], path_ids)
            fig.add_trace(go.Scatter3d(
                x=X[rest], y=Y[rest], z=Z[rest], mode="markers", hoverinfo="skip",
                showlegend=False,
                marker=dict(size=max(2, dot_size * 0.5), color="#bbbbbb", opacity=0.12),
            ))

        if ups:
            fig.add_trace(edge_trace([(selected, t) for t in ups], "Next step up", "#111111", "solid"))
        if downs:
            fig.add_trace(edge_trace([(s, selected) for s in downs], "Leads into this job", "#777777", "dash"))

        for level in LEVELS:
            ids = [j for j in ups + downs if LEVEL_TXT[j] == level]
            if ids:
                fig.add_trace(dot_trace(ids, level, size=dot_size + 6, opacity=1.0, outline=2))

        sel_level = LEVEL_TXT[selected]
        sel = dot_trace([selected], sel_level, size=dot_size + 11, opacity=1.0, outline=3,
                        name="Selected job")
        sel.marker.symbol = "diamond"
        fig.add_trace(sel)

        if "labels" in options:
            fig.add_trace(label_trace(selected, ups + downs))
            fig.add_trace(go.Scatter3d(
                x=[X[selected]], y=[Y[selected]], z=[Z[selected]], mode="text",
                text=[shorten(TITLES[selected], 40)], textposition="top center",
                textfont=dict(size=16, color="black"), hoverinfo="skip", showlegend=False,
            ))

        # zoom the x/y axes to the path so it fills the screen
        px_, py_ = X[path_ids], Y[path_ids]
        pad = max(1.0, 0.35 * max(np.ptp(px_), np.ptp(py_)))
        x_range = [float(px_.min() - pad), float(px_.max() + pad)]
        y_range = [float(py_.min() - pad), float(py_.max() + pad)]

    axis = dict(showbackground=False, showgrid=True, gridcolor="#e6e6e6", zeroline=False,
                showticklabels=False, title="")
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range, **axis),
            yaxis=dict(range=y_range, **axis),
            zaxis=dict(
                title="Career level", range=[0.6, 5.4],
                tickvals=list(range(1, 6)), ticktext=[LEVEL_SHORT[l] for l in LEVELS],
                tickfont=dict(size=13), showbackground=False, gridcolor="#d9d9d9", zeroline=False,
            ),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.8),
            camera=CAMERAS[view],
        ),
        legend=dict(itemclick=False, itemdoubleclick=False, font=dict(size=13),
                    bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="white",
        hoverlabel=dict(font_size=14),
        # the camera is kept while you tweak controls, and re-framed when the selection/view changes
        uirevision=f"{selected}|{view}",
    )
    return fig


# ----------------------------------------------------------------------
# Side panel (the always-legible version of the path)
# ----------------------------------------------------------------------
CARD = {"border": "1px solid #e3e3e3", "borderRadius": "8px", "padding": "8px 10px",
        "background": "white", "textAlign": "left", "width": "100%", "cursor": "pointer",
        "marginBottom": "6px", "fontSize": "14px"}


def level_tag(job):
    return html.Div([
        html.Span("●", style={"color": LEVEL_COLOR[LEVEL_TXT[job]], "marginRight": "6px"}),
        html.Span(LEVEL_SHORT[LEVEL_TXT[job]], style={"color": "#666", "fontSize": "12px"}),
    ])


def job_button(job):
    return html.Button(
        [html.Div(TITLES[job], style={"fontWeight": "600"}), level_tag(job)],
        id={"type": "nbr", "index": int(job)}, n_clicks=0, style=CARD,
    )


def build_panel(selected, direction, k):
    if selected is None:
        return [
            html.H3("How to explore", style={"marginTop": 0}),
            html.Ol([
                html.Li("Click any dot, or search for a job title above."),
                html.Li("The map zooms in on that job's career connections."),
                html.Li("Click a job in this panel (or a dot) to walk up or down the ladder."),
            ], style={"paddingLeft": "18px", "lineHeight": "1.6"}),
            html.P("Height = career level. Dots that are close together have similar job descriptions.",
                   style={"color": "#666"}),
        ]

    ups, downs = get_neighbors(selected, direction, k)
    desc = DESCRIPTIONS[selected]
    children = [
        html.Div("SELECTED JOB", style={"fontSize": "11px", "letterSpacing": "1px", "color": "#888"}),
        html.H3(TITLES[selected], style={"margin": "2px 0 4px 0"}),
        level_tag(selected),
    ]
    if desc:
        children.append(html.P(desc[:260] + ("…" if len(desc) > 260 else ""),
                               style={"color": "#555", "fontSize": "13px", "marginTop": "8px"}))

    children.append(html.H4(f"Next steps up ({len(ups)})", style={"marginBottom": "6px"}))
    children += [job_button(j) for j in ups] or [html.P("None shown.", style={"color": "#888"})]
    children.append(html.H4(f"Roles that lead here ({len(downs)})", style={"marginBottom": "6px"}))
    children += [job_button(j) for j in downs] or [html.P("None shown.", style={"color": "#888"})]
    return children


# ----------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------
def control(label, component, width="auto"):
    return html.Div(
        [html.Div(label, style={"fontSize": "12px", "color": "#666", "marginBottom": "2px"}), component],
        style={"width": width, "minWidth": width if width != "auto" else None},
    )


search_options = [
    {"label": f"{TITLES[i]} — {LEVEL_SHORT[LEVEL_TXT[i]]}", "value": int(i)}
    for i in sorted(np.where(has_level)[0], key=lambda i: TITLES[i].lower())
]

app = Dash(__name__)
app.title = "Career Paths Map"

app.layout = html.Div(
    style={"fontFamily": "Inter, Segoe UI, Arial, sans-serif", "padding": "12px 16px", "background": "#fafafa"},
    children=[
        dcc.Store(id="selected", data=None),
        html.Div([
            html.H2("Career Paths Map", style={"margin": 0}),
            html.Div("Explore how NYC jobs connect from student to executive",
                     style={"color": "#666", "fontSize": "14px"}),
        ]),
        html.Div(
            style={"display": "flex", "flexWrap": "wrap", "gap": "18px", "alignItems": "flex-end",
                   "margin": "12px 0"},
            children=[
                control("Search a job", dcc.Dropdown(id="search", options=search_options,
                                                     placeholder="Type a job title…"), "320px"),
                control("Show connections", dcc.RadioItems(
                    id="direction", value="both", inline=True,
                    options=[{"label": " Both ", "value": "both"},
                             {"label": " Up ", "value": "up"},
                             {"label": " Down ", "value": "down"}]), "190px"),
                control("Connections per direction",
                        dcc.Slider(id="k", min=1, max=10, step=1, value=3,
                                   marks={1: "1", 5: "5", 10: "10"}), "190px"),
                control("Dot size",
                        dcc.Slider(id="dot-size", min=2, max=12, step=1, value=5,
                                   marks={2: "2", 7: "7", 12: "12"}), "160px"),
                control("View", dcc.RadioItems(
                    id="view", value="3d", inline=True,
                    options=[{"label": " 3D ", "value": "3d"},
                             {"label": " Side ", "value": "side"},
                             {"label": " Top ", "value": "top"}]), "180px"),
                control("Career levels (overview)", dcc.Checklist(
                    id="level-filter", value=LEVELS, inline=True,
                    options=[{"label": f" {LEVEL_SHORT[l]} ", "value": l} for l in LEVELS]), "auto"),
                control("Options", dcc.Checklist(
                    id="options", value=["labels"], inline=True,
                    options=[{"label": " Labels ", "value": "labels"},
                             {"label": " Show faded background dots ", "value": "ghost"}]), "auto"),
                html.Button("Reset", id="clear-btn", n_clicks=0,
                            style={"padding": "8px 16px", "borderRadius": "6px", "cursor": "pointer"}),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "14px"},
            children=[
                dcc.Graph(
                    id="career-graph", style={"flex": "1", "height": "78vh", "background": "white",
                                              "border": "1px solid #e3e3e3", "borderRadius": "10px"},
                    config={"displaylogo": False, "scrollZoom": True},
                ),
                html.Div(id="panel", style={
                    "width": "340px", "height": "78vh", "overflowY": "auto", "background": "white",
                    "border": "1px solid #e3e3e3", "borderRadius": "10px", "padding": "14px"}),
            ],
        ),
        html.Div("Drag to rotate · scroll to zoom · use the toolbar's Pan tool (or right-drag) to move around",
                 style={"color": "#888", "fontSize": "12px", "marginTop": "6px"}),
    ],
)


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------
@app.callback(
    Output("selected", "data"),
    Input("career-graph", "clickData"),
    Input("search", "value"),
    Input("clear-btn", "n_clicks"),
    Input({"type": "nbr", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_job(click_data, search_value, _clear, _nbr_clicks):
    trigger = ctx.triggered_id

    if trigger == "clear-btn":
        return None

    if trigger == "search":
        return None if search_value is None else int(search_value)

    if trigger == "career-graph":
        if not click_data:
            return no_update
        custom = click_data["points"][0].get("customdata")
        if custom is None:      # clicked a label/edge, not a dot
            return no_update
        return int(custom[0] if isinstance(custom, (list, tuple)) else custom)

    if isinstance(trigger, dict):  # a job in the side panel
        if not ctx.triggered[0]["value"]:   # fires with 0 when the panel is rebuilt
            return no_update
        return int(trigger["index"])

    return no_update


@app.callback(Output("search", "value"), Input("clear-btn", "n_clicks"), prevent_initial_call=True)
def clear_search(_):
    return None


@app.callback(
    Output("career-graph", "figure"),
    Output("panel", "children"),
    Input("selected", "data"),
    Input("level-filter", "value"),
    Input("direction", "value"),
    Input("k", "value"),
    Input("dot-size", "value"),
    Input("options", "value"),
    Input("view", "value"),
)
def render(selected, level_filter, direction, k, dot_size, options, view):
    fig = build_figure(selected, level_filter or [], direction, k, dot_size, options or [], view)
    return fig, build_panel(selected, direction, k)


if __name__ == "__main__":
    app.run(debug=True)
