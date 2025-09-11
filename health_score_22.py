#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HT 22 / 33 kV cable-health dashboard — FULL-YEAR CONFUSION MATRIX ONLY

- Upload scored CSV (health_score, health_score_10, health_band, SWITCH_ID, etc.)
- (Optional) Upload faults CSV with TIME_OUTAGE; pick the year to compute ACTUAL_FAIL_YEAR
- Evaluation scope controls are retained:
    • Exclude 'Good' band from metrics
    • Hide 'Good' in graph & table
    • Apply current view filters to metrics
"""

import csv, tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import networkx as nx
from pyvis.network import Network
from typing import Dict, Optional
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
)

# ═════════════════════════════════════════════════╗
# 0) THEMES & CSS
# ═════════════════════════════════════════════════╝

THEMES = {
    "Dark": {
        "bg":  "#0b1220", "bg2": "#0f172a", "card": "#111827",
        "text": "#e5e7eb", "muted": "#a1a1aa", "primary": "#22c55e",
        "node": "#60a5fa", "good": "#22c55e", "mod": "#f59e0b", "poor": "#ef4444",
    },
    "Light": {
        "bg":  "#f5f7fb", "bg2": "#ffffff", "card": "#ffffff",
        "text": "#0f172a", "muted": "#475569", "primary": "#059669",
        "node": "#2563eb", "good": "#16a34a", "mod": "#d97706", "poor": "#b91c1c",
    },
}

def inject_theme_css(t: Dict[str, str]) -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {t['bg']} 0%, {t['bg2']} 100%);
            color: {t['text']};
        }}
        .block-container {{ padding-top: 1rem; padding-bottom: 2rem; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {t['card']};
            color: {t['text']};
            border-radius: 8px 8px 0 0;
        }}
        .stTabs div[role="tabpanel"] {{
            background-color: {t['card']};
            padding: 0.8rem 1rem 1rem 1rem;
            border-radius: 0 8px 8px 8px;
        }}
        div[data-testid="stMetricValue"] {{ color: {t['primary']}; }}
        .stDataFrame thead tr th {{
            background-color: {t['card']} !important; color: {t['text']} !important;
        }}
        iframe {{ background: transparent !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ═════════════════════════════════════════════════╗
# 1) HELPERS
# ═════════════════════════════════════════════════╝

def sniff_sep(buf: bytes) -> str:
    sample = buf[:10_000].decode("utf-8", errors="ignore")
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return ","

def norm_switch(series: pd.Series) -> pd.Series:
    s = (series.astype(str).str.upper().str.strip()
         .str.replace(r"^(SWNO_|SWNO|SW|S)\s*", "", regex=True)
         .str.replace(r"\D+", "", regex=True)
         .replace("", np.nan))
    return pd.to_numeric(s, errors="coerce").astype("Int64")

@st.cache_data(show_spinner=False)
def load_scored(upload) -> pd.DataFrame:
    raw = upload.read(); upload.seek(0)
    df = pd.read_csv(upload, sep=sniff_sep(raw), low_memory=False)
    df = df.rename(columns=lambda c:
                   "SOURCE_SS"      if str(c).startswith("SOURCE_SS") else
                   "DESTINATION_SS" if str(c).startswith("DESTINATION_SS") else str(c))
    df = df.loc[:, ~df.columns.duplicated()]
    need = {"health_score", "health_score_10", "SWITCH_ID", "health_band"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError("CSV missing columns: " + ", ".join(sorted(miss)))
    return df

# (set after theme)
BAND_COLORS: Dict[str, str] = {}

def ensure_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Edge_BandColor"] = out["health_band"].map(BAND_COLORS).fillna("#808080")
    return out

def compute_actual_fail_full_year(
    faults_df: pd.DataFrame,
    year: int,
    switch_col_candidates=("TO_SWITCH", "SWITCH_ID"),
    time_col="TIME_OUTAGE",
) -> pd.Series:
    """Return SWITCH_IDs with ≥1 fault anywhere in `year` (tz-safe)."""
    cols = {c.upper() for c in faults_df.columns}
    sw_col = next((c for c in switch_col_candidates if c.upper() in cols), None)
    if sw_col is None or time_col not in faults_df.columns:
        return pd.Series([], dtype="Int64")

    tmp = faults_df.copy()
    tmp["SWITCH_ID"] = norm_switch(tmp[sw_col])
    tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce", utc=True).dt.tz_localize(None)

    mask = (tmp[time_col].dt.year == year)
    return tmp.loc[mask, "SWITCH_ID"].dropna().astype("Int64")

def plot_health_distribution(scores: pd.Series, theme: Dict[str, str]):
    scores = pd.to_numeric(scores, errors="coerce").dropna().clip(0, 100)
    bins = np.arange(0, 101, 5)
    counts, edges = np.histogram(scores, bins=bins)
    fig, ax = plt.subplots(figsize=(5, 2.3), dpi=160)
    x = edges[:-1]
    colors = [BAND_COLORS["Poor"] if e < 40
              else BAND_COLORS["Moderate"] if e < 60
              else BAND_COLORS["Good"] for e in x]
    ax.bar(x + 2.5, counts, width=4.6, color=colors, edgecolor="#222", linewidth=0.4)
    ax.axvline(40, color=BAND_COLORS["Poor"], linestyle="--", linewidth=1, alpha=0.9)
    ax.axvline(60, color=BAND_COLORS["Moderate"], linestyle="--", linewidth=1, alpha=0.9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.35)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 100])
    ax.set_xlabel("Health score"); ax.set_ylabel("Count")
    fig.patch.set_facecolor(theme["card"]); ax.set_facecolor(theme["card"])
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_color(theme["text"])
    ax.xaxis.label.set_color(theme["muted"]); ax.yaxis.label.set_color(theme["muted"])
    ymax = max(counts.max(), 1)
    ax.text(20, ymax*0.95, "Poor", color=BAND_COLORS["Poor"], ha="center", va="top", fontsize=9)
    ax.text(55, ymax*0.95, "Moderate", color=BAND_COLORS["Moderate"], ha="center", va="top", fontsize=9)
    ax.text(85, ymax*0.95, "Good", color=BAND_COLORS["Good"], ha="center", va="top", fontsize=9)
    fig.tight_layout()
    return fig

def safe_metrics(y_true: np.ndarray, y_pred_bin: np.ndarray, cont_score: np.ndarray):
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc  = (tp + tn) / cm.sum() if cm.sum() else np.nan
    prec = precision_score(y_true, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true, y_pred_bin, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0
    f1   = f1_score(y_true, y_pred_bin, zero_division=0)
    try:
        auc = roc_auc_score(y_true, cont_score)
    except Exception:
        auc = np.nan
    return cm, tn, fp, fn, tp, acc, prec, rec, spec, f1, auc

# ═════════════════════════════════════════════════╗
# 2) PAGE & THEME
# ═════════════════════════════════════════════════╝

st.set_page_config(page_title="HT cable-health dashboard — Full Year", layout="wide")
with st.sidebar:
    st.markdown("### Appearance")
    theme_choice = st.selectbox("Theme", options=list(THEMES.keys()), index=0)
theme = THEMES[theme_choice]
inject_theme_css(theme)
BAND_COLORS = {"Poor": theme["poor"], "Moderate": theme["mod"], "Good": theme["good"]}

st.title("HT 22 / 33 kV Cable-Health Dashboard ")

# ═════════════════════════════════════════════════╗
# 3) LOAD SCORED CSV
# ═════════════════════════════════════════════════╝

file_scored = st.file_uploader("Upload **cable_health_scored.csv** (full-year scored)", type=["csv"])
if not file_scored:
    st.stop()

try:
    df_scored = load_scored(file_scored)
except Exception as e:
    st.error(str(e)); st.stop()

df_scored = ensure_helper_columns(df_scored)
st.session_state["df_base"] = df_scored.copy()   # no labels yet

# ═════════════════════════════════════════════════╗
# 4) OPTIONAL FAULTS CSV → PICK YEAR & LABEL FULL YEAR
# ═════════════════════════════════════════════════╝

st.write("---")
file_faults = st.file_uploader(
    "(optional) Upload raw faults CSV (with TIME_OUTAGE); pick year to compute ACTUAL_FAIL_YEAR",
    type=["csv"]
)

ft = None
selected_year: Optional[int] = None
df_year = st.session_state["df_base"].copy()

if file_faults:
    raw = file_faults.read(); file_faults.seek(0)
    ft = pd.read_csv(file_faults, sep=sniff_sep(raw), low_memory=False)

    if "TIME_OUTAGE" not in ft.columns:
        st.warning("Faults CSV must contain TIME_OUTAGE column.")
        ft = None
    else:
        ft["TIME_OUTAGE"] = pd.to_datetime(ft["TIME_OUTAGE"], errors="coerce", utc=True).dt.tz_localize(None)
        years = sorted([int(y) for y in ft["TIME_OUTAGE"].dt.year.dropna().unique().tolist()])
        if years:
            default_idx = years.index(2024) if 2024 in years else len(years)-1
            selected_year = st.selectbox("Faults year", options=years, index=default_idx)
            ids_year = compute_actual_fail_full_year(ft, year=selected_year)
            df_year["ACTUAL_FAIL_YEAR"] = df_year["SWITCH_ID"].isin(set(ids_year.tolist())).astype(int)
        else:
            st.warning("Could not parse any year values from TIME_OUTAGE.")

st.session_state["df_year"] = df_year.copy()

# ═════════════════════════════════════════════════╗
# 5) SIDEBAR FILTERS (affect graph & table) + EVALUATION SCOPE
# ═════════════════════════════════════════════════╝

with st.sidebar:
    st.header("Filters")
    d_rng  = st.slider("Decile (1 worst → 10 best)", 1, 10, (1,10))

    band_values = list(pd.unique(df_scored["health_band"].dropna()))
    order = [b for b in ["Good", "Moderate", "Poor"] if b in band_values]
    rest  = [b for b in band_values if b not in order]
    default_bands = order + rest
    b_sel  = st.multiselect("Band (from file)", default_bands, default=default_bands)

    kw     = st.text_input("Keyword (station contains)", placeholder="search…").upper()

    swno_selected = []
    if "SWNO" in df_scored.columns:
        swno_all = pd.to_numeric(df_scored["SWNO"], errors="coerce").dropna().astype(int).unique()
        swno_all = sorted(swno_all.tolist())
        swno_options = ["All"] + swno_all
        swno_selected = st.multiselect("Feeder IDs (SWNO)", options=swno_options, default=["All"],
                                       help="Choose 'All' to include every SWNO; otherwise pick specific feeders.")

    st.write("---")
    st.subheader("Evaluation scope")
    exclude_healthy_metrics = st.checkbox("Exclude 'Good' band from metrics", value=False)
    hide_healthy_view = st.checkbox("Hide 'Good' in graph & table", value=False)
    apply_filters_to_metrics = st.checkbox("Apply current view filters to metrics", value=False)

# View mask (graph & table)
df_for_view = st.session_state["df_year"].copy()
mask = (df_for_view["health_score_10"].between(*d_rng) & df_for_view["health_band"].isin(b_sel))
if kw:
    for col in ("SOURCE_SS","DESTINATION_SS"):
        if col not in df_for_view.columns: df_for_view[col] = ""
    mask &= (df_for_view["SOURCE_SS"].astype(str).str.upper().str.contains(kw) |
             df_for_view["DESTINATION_SS"].astype(str).str.upper().str.contains(kw))
if "SWNO" in df_for_view.columns and swno_selected and "All" not in swno_selected:
    swno_norm = pd.to_numeric(df_for_view["SWNO"], errors="coerce").astype("Int64")
    mask &= swno_norm.isin(pd.Series(swno_selected, dtype="Int64"))
if hide_healthy_view:
    mask &= df_for_view["health_band"].ne("Good")

df_view = df_for_view[mask].copy()
if df_view.empty:
    st.warning("No rows match your filters."); st.stop()

# Metrics base (full-year)
metrics_base_full_year = df_view.copy() if apply_filters_to_metrics else st.session_state["df_year"].copy()
if exclude_healthy_metrics:
    metrics_base_full_year = metrics_base_full_year[metrics_base_full_year["health_band"] != "Good"]

# ═════════════════════════════════════════════════╗
# 6) NETWORK GRAPH (edge shows Health Score)
# ═════════════════════════════════════════════════╝

colour_col = "Edge_BandColor"
G = nx.MultiDiGraph()
for _, row in df_view.iterrows():
    src = row.get("SOURCE_SS", "UNKNOWN_SRC")
    dst = row.get("DESTINATION_SS", "UNKNOWN_DST")
    d = int(pd.to_numeric(row.get("health_score_10", 0), errors="coerce") or 0)
    hs_val = pd.to_numeric(row.get("health_score"), errors="coerce")
    hs = int(hs_val) if pd.notna(hs_val) else None
    edge_label = f"HS {hs}" if hs is not None else "HS -"
    tooltip = "\n".join([
        f"Band : {row.get('health_band','-')}",
        f"Health-score : {hs if hs is not None else '-'}",
        f"Decile : {d}",
        f"Primary driver : {row.get('primary_health_driver','-')}",
        f"Top3 drivers : {row.get('top3_health_drivers','-')}",
        f"SWNO : {row.get('SWNO','-')}",
        f"Length (m) : {row.get('LENGTH_M','-')}",
    ])
    for node in (src, dst):
        if node not in G:
            G.add_node(node, title=f"Sub-station : {node}", color=theme["node"])
    G.add_edge(
        src, dst,
        color=row.get(colour_col, "#EBDDDD"),
        width=3 + (10 - d),
        label=edge_label,
        title=tooltip
    )

net = Network(height="900px", bgcolor="rgba(0,0,0,0)", directed=True)
net.from_nx(G)
net.set_options("""
var options = {
  "nodes": {"size": 40},
  "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
            "smooth": {"type": "dynamic"}},
  "physics": {
    "forceAtlas2Based": {"gravitationalConstant": -80,
                         "centralGravity": 0.005,
                         "springLength": 180,
                         "springConstant": 0.08},
    "solver": "forceAtlas2Based",
    "timestep": 0.6,
    "stabilization": {"enabled": true, "iterations": 120}}
}""")

with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
    net.save_graph(tmp.name)
    html = open(tmp.name, encoding="utf-8").read() + """
    <script> setTimeout(()=>{ if(window.network) network.setOptions({physics:false}); }, 30000); </script>"""
    st.components.v1.html(html, height=900, scrolling=True)

# ═════════════════════════════════════════════════╗
# 7) TABS — Summary, Full-year CM, Data
# ═════════════════════════════════════════════════╝

tab1, tab2, tab3 = st.tabs(["Summary", "Full-year confusion matrix", "Data (filtered)"])

with tab1:
    st.metric("Cables shown", len(df_view))
    st.metric("Mean decile", f"{df_view['health_score_10'].mean():.2f}")
    st.markdown("#### Health-score distribution (filtered)")
    fig = plot_health_distribution(df_view["health_score"], theme)
    st.pyplot(fig, use_container_width=False)

    worst = df_view.sort_values("health_score", ascending=True)
    if not worst.empty:
        w = worst.iloc[0]
        st.metric("Worst cable (filtered)",
                  f"SWNO {w.get('SWNO','-')}",
                  help=f"Health-score = {w['health_score']}, Band = {w['health_band']}")
        show_cols = [c for c in ["SWNO","SOURCE_SS","DESTINATION_SS","health_score","health_band",
                                 "primary_health_driver","top3_health_drivers","MEASUREDLENGTH","LENGTH_M"]
                     if c in worst.columns]
        st.markdown("**Top 10 worst cables (lowest health_score)**")
        st.dataframe(worst[show_cols].head(10))

with tab2:
    df_eval = metrics_base_full_year.copy()
    if "ACTUAL_FAIL_YEAR" not in df_eval.columns:
        st.info("Upload a faults CSV and pick the **Faults year** to enable the full-year confusion matrix.")
    else:
        n_before = (df_view.copy() if apply_filters_to_metrics else st.session_state["df_year"]).shape[0]
        n_after  = df_eval.shape[0]
        st.caption(
            f"Evaluating on {n_after} cables (out of {n_before}). "
            + ("Filters applied to metrics. " if apply_filters_to_metrics else "Using full set for metrics. ")
            + ("Excluding Good band." if exclude_healthy_metrics else "")
        )

        if df_eval.empty:
            st.warning("No cables left to evaluate after exclusions/filters.")
        else:
            y_true = df_eval["ACTUAL_FAIL_YEAR"].astype(int).values
            y_pred_bin = df_eval["health_band"].map({"Poor":1, "Moderate":1, "Good":0}).astype(int).values
            cont = 1 - df_eval["health_score"].values/100
            cm, tn, fp, fn, tp, acc, prec, rec, spec, f1, auc = safe_metrics(y_true, y_pred_bin, cont)

            fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Pred OK","Pred Fail"],
                        yticklabels=["Actual OK","Actual Fail"], ax=ax)
            fig.patch.set_facecolor(theme["card"]); ax.set_facecolor(theme["card"])
            ax.set_title(f"Full-year — {selected_year if selected_year else '(year not set)'}", color=theme["text"])
            ax.set_xlabel("Prediction"); ax.set_ylabel("Reality")
            for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_color(theme["text"])
            ax.xaxis.label.set_color(theme["muted"]); ax.yaxis.label.set_color(theme["muted"])
            st.pyplot(fig, use_container_width=False)

            st.markdown(f"""
**TP** = {tp}  | **FP** = {fp}  | **FN** = {fn}  | **TN** = {tn}

| Metric | Value |
|--------|-------|
| **Accuracy**    | `{acc:.2%}` |
| **Precision**   | `{prec:.2%}` |
| **Recall**      | `{rec:.2%}` |
| **Specificity** | `{spec:.2%}` |
| **F1-score**    | `{f1:.2%}` |
| **AUROC**       | `{auc:.3f}` |
""")

with tab3:
    cols = ["SWNO","SOURCE_SS","DESTINATION_SS","health_score_10",
            "health_band","health_score","primary_health_driver","top3_health_drivers","pred_faults_2024"]
    cols = [c for c in cols if c in df_view.columns]
    st.dataframe(df_view.sort_values("health_score_10", ascending=True)[cols])

# ═════════════════════════════════════════════════╗
# 8) LEGEND
# ═════════════════════════════════════════════════╝
st.markdown("### Legend")
band_labels = [("Good", BAND_COLORS["Good"]),
               ("Moderate", BAND_COLORS["Moderate"]),
               ("Poor", BAND_COLORS["Poor"])]
fig, ax = plt.subplots(figsize=(4, 0.8), dpi=160)
for idx, (label, col) in enumerate(band_labels):
    ax.bar(idx, 1, color=col)
ax.set_xticks(range(len(band_labels)))
ax.set_xticklabels([b for b,_ in band_labels], fontsize=9, color=theme["text"])
ax.set_yticks([])
fig.patch.set_facecolor(theme["card"]); ax.set_facecolor(theme["card"])
for spine in ("top","right","left","bottom"): ax.spines[spine].set_visible(False)
st.pyplot(fig, use_container_width=True)
