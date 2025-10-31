import base64, io, re, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, State
from dash.dcc import send_string

# ---------------- Load artifacts ----------------
ART = Path("artifacts")
REG = joblib.load(ART / "model_reg.pkl")
CLF = joblib.load(ART / "model_clf.pkl")
FEATURES = json.loads((ART / "features.json").read_text())["features"]
DEPLOY = json.loads((ART / "deploy_meta.json").read_text())
QUANTILES = json.loads((ART / "feature_quantiles.json").read_text()) if (ART / "feature_quantiles.json").exists() else {}

PI_REG = None
PI_CLF = None
try:
    PI_REG = pd.read_csv(ART / "importance_reg_permutation.csv")
except Exception:
    pass
try:
    PI_CLF = pd.read_csv(ART / "importance_clf_permutation.csv")
except Exception:
    pass

# ---------------- Inline theme ----------------
COLORS = {
    "bg": "#0f172a",            # slate-900
    "panel": "#111827",         # gray-900
    "card": "#1f2937",          # gray-800
    "card_alt": "#111827",
    "text": "#e5e7eb",          # gray-200
    "muted": "#9ca3af",         # gray-400
    "accent": "#22c55e",        # green-500
    "accent2": "#60a5fa",       # blue-400
    "warn": "#f59e0b",          # amber-500
    "danger": "#ef4444",        # red-500
    "border": "#374151",        # gray-700
}

PAGE = {
    "backgroundColor": COLORS["bg"],
    "color": COLORS["text"],
    "fontFamily": "Segoe UI, Inter, system-ui, -apple-system, Arial, sans-serif",
    "padding": "18px 22px",
}

HEADER = {
    "margin": "0 0 6px 0",
    "fontSize": "28px",
    "fontWeight": "700",
}

SUBTITLE = {
    "margin": "0 0 18px 0",
    "color": COLORS["muted"],
    "fontSize": "14px",
}

TABS = {
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "backgroundColor": COLORS["panel"],
    "padding": "0",
}
TAB_STYLE = {
    "padding": "10px 14px",
    "border": "0",
    "backgroundColor": COLORS["panel"],
    "color": COLORS["muted"],
}
TAB_SELECTED = {
    "padding": "10px 14px",
    "border": "0",
    "borderBottom": f"2px solid {COLORS['accent2']}",
    "backgroundColor": COLORS["panel"],
    "color": COLORS["text"],
    "fontWeight": 600,
}

SECTION = {
    "backgroundColor": COLORS["panel"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "14px",
    "padding": "16px",
    "margin": "14px 0",
}

CONTROL_LABEL = {
    "display": "block",
    "fontSize": "12px",
    "color": COLORS["muted"],
    "marginBottom": "6px",
}

NUM_INPUT = {
    "width": "220px",
    "backgroundColor": COLORS["card"],
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "padding": "8px 10px",
    "outline": "none",
}

BTN_PRIMARY = {
    "backgroundColor": COLORS["accent2"],
    "color": "#0b1220",
    "border": "0",
    "borderRadius": "10px",
    "padding": "10px 16px",
    "fontWeight": 600,
    "cursor": "pointer",
    "boxShadow": "0 6px 16px rgba(96,165,250,0.25)",
}

BTN_SECONDARY = {
    "backgroundColor": COLORS["card"],
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "10px",
    "padding": "10px 14px",
    "cursor": "pointer",
}

METRIC_CARD = {
    "backgroundColor": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "14px",
    "padding": "14px",
    "width": "31%",
    "display": "inline-block",
    "verticalAlign": "top",
}

METRIC_LABEL = {
    "color": COLORS["muted"],
    "fontSize": "12px",
    "marginBottom": "6px",
}
METRIC_VALUE = {
    "fontSize": "22px",
    "fontWeight": 700,
}

DIVIDER = {"borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}

TABLE_STYLE = {
    "width": "100%",
    "backgroundColor": COLORS["card"],
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "overflow": "hidden",
}

# DataTable style config
DT_STYLE = dict(
    style_table={"overflowX": "auto", "backgroundColor": COLORS["card"]},
    style_header={
        "backgroundColor": COLORS["card_alt"],
        "color": COLORS["text"],
        "border": f"1px solid {COLORS['border']}",
        "fontWeight": "600",
    },
    style_cell={
        "backgroundColor": COLORS["card"],
        "color": COLORS["text"],
        "border": f"1px solid {COLORS['border']}",
        "fontSize": "13px",
    },
)

# ---------------- Helpers ----------------
def ensure_df_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]

def predict_df(df_in: pd.DataFrame) -> pd.DataFrame:
    X = ensure_df_cols(df_in, FEATURES)
    y_log = REG.predict(X)
    owners_pred = np.expm1(y_log)
    if hasattr(CLF, "predict_proba"):
        p_succ = CLF.predict_proba(X)[:, 1]
    else:
        z = CLF.decision_function(X)
        p_succ = 1.0 / (1.0 + np.exp(-z))
    out = df_in.copy()
    out["owners_pred"] = owners_pred
    out["success_prob"] = p_succ
    if "price" in out.columns:
        out["revenue_proxy"] = out["price"] * out["owners_pred"]
    return out

def quantile_bounds(c: str) -> tuple[float, float, float]:
    q = QUANTILES.get(c, {})
    q01 = float(q.get("0.01", 0.0))
    q50 = float(q.get("0.5",  0.0))
    q95 = float(q.get("0.95", 1.0))
    lo = min(q01, q50, q95)
    hi = max(q01, q50, q95)
    if lo == hi:
        lo, hi = 0.0, (100.0 if ("discount" in c or "userscore" in c) else 1_000.0)
    return lo, q50, hi

def id_for_feature(f: str) -> str:
    return "input_" + re.sub(r"[^0-9A-Za-z_]+", "_", f)

INPUT_IDS = {f: id_for_feature(f) for f in FEATURES}

def default_inputs() -> dict:
    d = {}
    for f in FEATURES:
        _, mid, _ = quantile_bounds(f)
        d[f] = float(mid)
    d["discount"]  = float(min(max(d.get("discount", 0.0), 0.0), 100.0))
    d["userscore"] = float(min(max(d.get("userscore", 75.0), 0.0), 100.0))
    return d

def build_heatmap(inputs: dict) -> go.Figure | None:
    if "price" not in inputs or "discount" not in inputs:
        return None
    base = pd.DataFrame([inputs])
    p0 = float(base["price"].iloc[0])
    p_vals = np.linspace(max(0.0, p0*0.5), p0*1.5, 15)
    d_vals = np.linspace(0, 90, 16)
    grid = pd.DataFrame([(p, d) for p in p_vals for d in d_vals], columns=["price","discount"])
    probe = base.drop(columns=["price","discount"], errors="ignore")
    sim = pd.concat([probe]*len(grid), ignore_index=True)
    sim[["price","discount"]] = grid.values
    scored = predict_df(sim)
    Z = scored["revenue_proxy"].values.reshape(len(p_vals), len(d_vals))
    fig = go.Figure(data=go.Heatmap(z=Z, x=d_vals, y=p_vals, colorbar=dict(title="Revenue proxy")))
    fig.update_layout(
        xaxis_title="Discount",
        yaxis_title="Price",
        margin=dict(l=16, r=16, t=26, b=16),
        height=420,
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
    )
    return fig

# ---------------- App ----------------
app = Dash(__name__)
server = app.server

# Controls for single scenario
def feature_control(f: str):
    lo, mid, hi = quantile_bounds(f)
    if "discount" in f:
        return html.Div([
            html.Label(f, style=CONTROL_LABEL),
            dcc.Slider(
                id=INPUT_IDS[f], min=0, max=100, step=1,
                value=float(np.clip(mid, 0.0, 100.0)),
                marks=None, tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ),
        ], style={"marginBottom": "14px"})
    if "userscore" in f:
        return html.Div([
            html.Label(f, style=CONTROL_LABEL),
            dcc.Slider(
                id=INPUT_IDS[f], min=0, max=100, step=0.5,
                value=float(np.clip(mid, 0.0, 100.0)),
                marks=None, tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
            ),
        ], style={"marginBottom": "14px"})
    # numeric
    return html.Div([
        html.Label(f, style=CONTROL_LABEL),
        dcc.Input(
            id=INPUT_IDS[f], type="number", value=float(mid),
            min=float(lo), max=float(hi), step=0.01,
            style=NUM_INPUT,
        ),
    ], style={"marginBottom": "14px"})

controls = [feature_control(f) for f in FEATURES]

app.layout = html.Div([
    html.H2("🎮 Steam KPI Scenario Planner — Dash", style=HEADER),
    html.Div("Predict owners and success probability. Single scenario, batch scoring, and insights.", style=SUBTITLE),

    html.Div([
        dcc.Tabs(style=TABS, children=[
            dcc.Tab(label="Single Scenario", style=TAB_STYLE, selected_style=TAB_SELECTED, children=[
                html.Div([
                    html.Div(controls, style={"columnCount": 3, "columnGap": "16px"}),
                    html.Div([
                        html.Button("Predict single scenario", id="predict-btn", style=BTN_PRIMARY),
                    ], style={"margin": "6px 0 10px 0"}),

                    html.Div([
                        html.Div([
                            html.Div("Owners (pred)", style=METRIC_LABEL),
                            html.Div(id="owners-metric", style=METRIC_VALUE),
                        ], style=METRIC_CARD),
                        html.Div([
                            html.Div("Success prob", style=METRIC_LABEL),
                            html.Div(id="success-metric", style=METRIC_VALUE),
                        ], style=METRIC_CARD),
                        html.Div([
                            html.Div("Decision", style=METRIC_LABEL),
                            html.Div(id="decision-badge", style=METRIC_VALUE),
                        ], style=METRIC_CARD),
                    ], style={"marginTop": "6px"}),

                    html.Div(style=DIVIDER),
                    html.H4("Price × Discount sensitivity", style={"margin": "0 0 8px 0"}),
                    html.Div([
                        dcc.Graph(id="heatmap-graph"),
                    ], style=SECTION),

                    html.Div(style=DIVIDER),
                    html.H4("Prediction table", style={"margin": "0 0 8px 0"}),
                    html.Div([
                        dash_table.DataTable(
                            id="single-table", page_size=5, **DT_STYLE
                        )
                    ], style=TABLE_STYLE),
                ], style=SECTION),
            ]),
            dcc.Tab(label="Batch Scoring", style=TAB_STYLE, selected_style=TAB_SELECTED, children=[
                html.Div([
                    html.P("Upload CSV. File must contain these columns: " + ", ".join(FEATURES), style={"color": COLORS["muted"]}),
                    html.Div([
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(["Drag & Drop or ", html.Span("Select CSV", style={"color": COLORS["accent2"], "textDecoration": "underline"})]),
                            multiple=False, accept=".csv",
                            style={
                                "width": "100%", "height": "80px", "lineHeight": "80px",
                                "borderWidth": "1px", "borderStyle": "dashed", "borderRadius": "10px",
                                "borderColor": COLORS["border"], "textAlign": "center", "margin": "8px 0",
                                "backgroundColor": COLORS["card"]
                            }
                        )
                    ]),
                    html.Div([
                        html.Button("Score file", id="score-btn", style=BTN_PRIMARY),
                        html.Button("Download predictions", id="download-btn", style={**BTN_SECONDARY, "marginLeft": "10px"}),
                        dcc.Download(id="download-preds"),
                    ], style={"margin": "6px 0 10px 0"}),

                    html.Div([
                        dash_table.DataTable(id="batch-table", page_size=15, **DT_STYLE),
                        dcc.Store(id="batch-csv-store"),
                    ], style=TABLE_STYLE),
                ], style=SECTION),
            ]),
            dcc.Tab(label="Insights", style=TAB_STYLE, selected_style=TAB_SELECTED, children=[
                html.Div([
                    html.H4("Global Importance (Permutation) — Regression", style={"margin": "0 0 8px 0"}),
                    dash_table.DataTable(
                        id="pi-reg",
                        data=(PI_REG.head(15).to_dict("records") if PI_REG is not None else []),
                        columns=([{"name": c, "id": c} for c in (PI_REG.columns if PI_REG is not None else ["feature", "mean_importance"])]),
                        page_size=15, **DT_STYLE
                    ),
                    html.Div(style=DIVIDER),
                    html.H4("Global Importance (Permutation) — Classification", style={"margin": "0 0 8px 0"}),
                    dash_table.DataTable(
                        id="pi-clf",
                        data=(PI_CLF.head(15).to_dict("records") if PI_CLF is not None else []),
                        columns=([{"name": c, "id": c} for c in (PI_CLF.columns if PI_CLF is not None else ["feature", "mean_importance"])]),
                        page_size=15, **DT_STYLE
                    ),
                    html.Div(style=DIVIDER),
                    html.H4("Deployed thresholds", style={"margin": "0 0 8px 0"}),
                    html.Pre(json.dumps(DEPLOY, indent=2), style={
                        "backgroundColor": COLORS["card"],
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "10px",
                        "padding": "12px",
                        "color": COLORS["text"],
                        "overflowX": "auto"
                    }),
                ], style=SECTION),
            ]),
        ])
    ], style={"maxWidth": "1200px", "margin": "0 auto"}),

    # Preload defaults to JS via Store
    dcc.Store(id="defaults-store", data=default_inputs()),
], style=PAGE)

# ------------- Callbacks -------------

@app.callback(
    Output("owners-metric","children"),
    Output("success-metric","children"),
    Output("decision-badge","children"),
    Output("single-table","data"),
    Output("single-table","columns"),
    Output("heatmap-graph","figure"),
    Input("predict-btn","n_clicks"),
    *[Input(INPUT_IDS[f], "value") for f in FEATURES],
    State("defaults-store","data"),
    prevent_initial_call=True
)
def do_predict(n_clicks, *args):
    vals = list(args)
    defaults = vals[-1]
    vals = vals[:-1]
    inputs = {}
    for f, v in zip(FEATURES, vals):
        if v is None:
            inputs[f] = float(defaults.get(f, 0.0))
        else:
            try:
                inputs[f] = float(v)
            except Exception:
                inputs[f] = float(defaults.get(f, 0.0))
    df = pd.DataFrame([inputs])
    pred = predict_df(df)
    owners = f"{pred['owners_pred'].iloc[0]:,.0f}"
    prob = float(pred["success_prob"].iloc[0])
    prob_txt = f"{prob:.3f}"
    thr = float(DEPLOY.get("threshold_gbc", 0.5))
    dec = ("PASS" if prob >= thr else "FAIL")
    table_data = pred.to_dict("records")
    table_cols = [{"name": c, "id": c} for c in pred.columns]
    fig = build_heatmap(inputs) or go.Figure(
        layout=dict(
            paper_bgcolor=COLORS["panel"], plot_bgcolor=COLORS["panel"],
            font=dict(color=COLORS["text"]), margin=dict(l=16, r=16, t=26, b=16), height=420
        )
    )
    return owners, prob_txt, dec, table_data, table_cols, fig

@app.callback(
    Output("batch-table","data"),
    Output("batch-table","columns"),
    Output("batch-csv-store","data"),
    Input("score-btn","n_clicks"),
    State("upload-data","contents"),
    State("upload-data","filename"),
    prevent_initial_call=True
)
def score_batch(n, contents, filename):
    if not contents:
        return [], [], ""
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.BytesIO(decoded))
        pred = predict_df(df)
        data = pred.head(200).to_dict("records")
        cols = [{"name": c, "id": c} for c in pred.columns]
        csv_txt = pred.to_csv(index=False)
        return data, cols, csv_txt
    except Exception as e:
        err = pd.DataFrame([{"error": f"Failed to score: {e}"}])
        return err.to_dict("records"), [{"name": "error", "id": "error"}], ""

@app.callback(
    Output("download-preds","data"),
    Input("download-btn","n_clicks"),
    State("batch-csv-store","data"),
    prevent_initial_call=True
)
def download_preds(n, csv_txt):
    if not csv_txt:
        return None
    return send_string(lambda: csv_txt, "predictions.csv")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8051, debug=False)
