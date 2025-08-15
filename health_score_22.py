#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HT 22 / 33 kV cable-health dashboard
Compatible with the “HT-cable Poisson-LSTM + 8-factor health-score (2018-2024)”

"""

import streamlit as st
import pandas as pd, numpy as np, csv, matplotlib.pyplot as plt
import seaborn as sns, networkx as nx
from pyvis.network import Network
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, recall_score, f1_score)
import tempfile
from typing import Optional

# ═════════════════════════════════════════════════╗
# 0. helper utilities                            ║
# ═════════════════════════════════════════════════╝

def sniff_sep(buf: bytes) -> str:
    sample = buf[:10_000].decode("utf-8", errors="ignore")
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return ","


def norm_switch(series: pd.Series) -> pd.Series:
    """Normalize various SW formats to Int64 (digits only)."""
    s = (series.astype(str)
               .str.upper().str.strip()
               .str.replace(r"^(SWNO_|SWNO|SW|S)\s*", "", regex=True)
               .str.replace(r"\D+", "", regex=True)
               .replace("", np.nan))
    return pd.to_numeric(s, errors="coerce").astype("Int64")


@st.cache_data(show_spinner=False)
def load_scored(upload) -> pd.DataFrame:
    raw = upload.read(); upload.seek(0)
    df = pd.read_csv(upload, sep=sniff_sep(raw), low_memory=False)
    # Canonicalize station column names if they have prefixes
    df = df.rename(columns=lambda c:
                   "SOURCE_SS"      if str(c).startswith("SOURCE_SS") else
                   "DESTINATION_SS" if str(c).startswith("DESTINATION_SS") else str(c))
    df = df.loc[:, ~df.columns.duplicated()]
    need = {"health_score", "health_score_10", "SWITCH_ID", "health_band"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError("CSV missing columns: " + ", ".join(sorted(miss)))
    return df


# ── high-contrast palettes (use health_band as-is) ──────────────────────
BAND_COLORS = {
    "Poor":     "#B22222",  # Firebrick (red)
    "Moderate": "#FF8C00",  # Dark Orange
    "Good":     "#228B22",  # Forest Green
}

def prob_colour(p: float) -> str:
    """Strong color for probability bands (0–1)."""
    p = float(np.clip(p, 0.0, 1.0))
    if p < 0.2:  return "#006400"  # Dark Green
    if p < 0.4:  return "#9ACD32"  # Yellow Green
    if p < 0.6:  return "#FFD700"  # Gold
    if p < 0.8:  return "#FF8C00"  # Dark Orange
    return "#B22222"               # Firebrick


def ensure_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create:
      - failure_prob: if missing → (100 - health_score)/100
      - Edge_BandColor: strong colors by health_band (from CSV)
      - Edge_Prob: strong colors by prob band
    """
    out = df.copy()

    # failure probability (if not provided)
    if "failure_prob" not in out.columns and "health_score" in out.columns:
        out["failure_prob"] = (100 - out["health_score"]) / 100.0

    # Colors (use health_band directly as provided in CSV)
    out["Edge_BandColor"] = out["health_band"].map(BAND_COLORS).fillna("#808080")
    if "failure_prob" in out.columns:
        out["Edge_Prob"] = out["failure_prob"].astype(float).apply(prob_colour)

    return out


def compute_actual_fail_2024(faults_df: pd.DataFrame,
                             switch_col_candidates=("TO_SWITCH","SWITCH_ID"),
                             time_col="TIME_OUTAGE") -> Optional[pd.Series]:
    """Return a Series of SWITCH_ID (Int64) that failed in 2024, or None if unavailable."""
    cols = set(faults_df.columns.str.upper())
    sw_col = None
    for c in switch_col_candidates:
        if c.upper() in cols:
            sw_col = c
            break
    if sw_col is None or time_col not in faults_df.columns:
        return None

    # Normalize switch id
    tmp = faults_df.copy()
    tmp["SWITCH_ID"] = norm_switch(tmp[sw_col])

    # Parse dates
    if not np.issubdtype(tmp[time_col].dtype, np.datetime64):
        with st.spinner("Parsing TIME_OUTAGE as datetime…"):
            tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")

    # Keep ONLY year 2024 faults
    tmp_24 = tmp[tmp[time_col].dt.year == 2024]
    if tmp_24.empty:
        return pd.Series([], dtype="Int64")
    return tmp_24["SWITCH_ID"].dropna().astype("Int64")


# ═════════════════════════════════════════════════╗
# 1. front-matter                                ║
# ═════════════════════════════════════════════════╝

st.set_page_config(page_title="HT cable-health dashboard", layout="wide")
st.title("HT 22 / 33 kV Cable-Health Dashboard  (1 = worst → 10 = best)")

file_scored = st.file_uploader("Upload **cable_health_2024_scored.csv**", type=["csv"])
if not file_scored:
    st.stop()

try:
    df = load_scored(file_scored)
except Exception as e:
    st.error(str(e)); st.stop()

# Keep an immutable copy of ALL rows (for metrics on full population)
st.session_state["df_all"] = df.copy()

# Ensure helper columns on BOTH frames
df = ensure_helper_columns(df)
st.session_state["df_all"] = ensure_helper_columns(st.session_state["df_all"])

# ═════════════════════════════════════════════════╗
# 2. optional fault CSV → ACTUAL_FAIL_24 (ONLY 2024) ║
# ═════════════════════════════════════════════════╝

st.write("---")
file_faults = st.file_uploader(
    "(optional) Upload raw faults CSV — we'll mark ACTUAL_FAIL_24 using ONLY faults from year 2024",
    type=["csv"]
)

if file_faults:
    raw = file_faults.read(); file_faults.seek(0)
    ft = pd.read_csv(file_faults, sep=sniff_sep(raw), low_memory=False)
    failed_switches_2024 = compute_actual_fail_2024(ft)

    if failed_switches_2024 is None:
        st.warning("Fault file lacks TIME_OUTAGE or switch column — metrics disabled.")
    else:
        # Mark ACTUAL_FAIL_24 on BOTH frames using only 2024 faults
        for _frame_name in ("df_all",):
            _frame = st.session_state[_frame_name].copy()
            _frame["ACTUAL_FAIL_24"] = _frame["SWITCH_ID"].isin(set(failed_switches_2024.dropna().tolist())).astype(int)
            st.session_state[_frame_name] = _frame

        # Also add to filtered working frame so Summary tab can show AUROC there if needed
        df["ACTUAL_FAIL_24"] = df["SWITCH_ID"].isin(set(failed_switches_2024.dropna().tolist())).astype(int)

# ═════════════════════════════════════════════════╗
# 3. sidebar filters (only affect graph + table)  ║
# ═════════════════════════════════════════════════╝

with st.sidebar:
    st.header("Filters")
    d_rng  = st.slider("Decile (1 worst → 10 best)", 1, 10, (1,10))
    # Build band list from CSV column exactly as-is
    band_values = list(pd.unique(df["health_band"].dropna()))
    # Keep order Good/Moderate/Poor if present
    order = [b for b in ["Good", "Moderate", "Poor"] if b in band_values]
    rest  = [b for b in band_values if b not in order]
    default_bands = order + rest
    b_sel  = st.multiselect("Band (from file)", default_bands, default=default_bands)
    kw     = st.text_input("Keyword (station contains)", placeholder="search…").upper()
    colour_mode = st.radio("Colour edges by", ("Band (health_band)", "Failure probability"), index=0)

mask = (
    df["health_score_10"].between(*d_rng) &
    df["health_band"].isin(b_sel)
)
if kw:
    for col in ("SOURCE_SS","DESTINATION_SS"):
        if col not in df.columns:
            df[col] = ""  # if absent in scored file
    mask &= (
        df["SOURCE_SS"].astype(str).str.upper().str.contains(kw) |
        df["DESTINATION_SS"].astype(str).str.upper().str.contains(kw)
    )

df_f = df[mask].copy()
if df_f.empty:
    st.warning("No rows match your filters."); st.stop()

# ═════════════════════════════════════════════════╗
# 4. build pyvis network from FILTERED view       ║
# ═════════════════════════════════════════════════╝

colour_col = {
    "Band (health_band)": "Edge_BandColor",
    "Failure probability": "Edge_Prob"
}[colour_mode]

G = nx.MultiDiGraph()
for _, row in df_f.iterrows():
    src = row.get("SOURCE_SS", "UNKNOWN_SRC")
    dst = row.get("DESTINATION_SS", "UNKNOWN_DST")
    d = int(row["health_score_10"])  # still use decile for width
    tooltip = "\n".join([
        f"Band : {row['health_band']}",
        f"Decile : {d}",
        f"Failure prob : {row['failure_prob']:.2%}" if "failure_prob" in row else "Failure prob : -",
        f"Primary driver : {row.get('primary_health_driver','-')}",
        f"Top3 drivers : {row.get('top3_health_drivers','-')}",
        f"SWNO : {row.get('SWNO','-')}",
        f"Length (m) : {row.get('LENGTH_M','-')}",
        f"Pred faults 2024 : {row.get('pred_faults_2024','-')}"
    ])
    for node in (src, dst):
        if node not in G:
            G.add_node(node, title=f"Sub-station : {node}", color="#6baed6")
    G.add_edge(src, dst,
               color=row.get(colour_col, "#808080"),
               width=3 + (10 - d),
            #    label=row["health_band"],
               label = row["decile"] if "decile" in row else str(d),
               font={"size": 14, "color": "#000000"},
               title=tooltip)

net = Network(height="900px", bgcolor="#fff", directed=True)
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
    <script>
      setTimeout(()=>{ if(window.network) network.setOptions({physics:false}); }, 30000);
    </script>"""
    st.components.v1.html(html, height=900, scrolling=True)

# ═════════════════════════════════════════════════╗
# 5. metrics tabs                                ║
# ═════════════════════════════════════════════════╝

(tab1, tab2, tab3) = st.tabs(["Summary", "Confusion matrix (ALL cables)", "Data (filtered)"])

# Tab 1: KPI summary (on filtered view; AUROC shown only if ACTUAL_FAIL_24 exists)
with tab1:
    st.metric("Cables shown", len(df_f))
    st.metric("Mean decile", f"{df_f['health_score_10'].mean():.2f}")
    if "failure_prob" in df_f.columns:
        st.metric("Mean failure prob", f"{df_f['failure_prob'].mean():.2%}")
    if "ACTUAL_FAIL_24" in df_f.columns:
        y_true = df_f["ACTUAL_FAIL_24"].astype(int)
        # continuous score: lower health_score => higher risk
        auc  = roc_auc_score(y_true, 1 - df_f["health_score"]/100)
        st.metric("AUROC (filtered view)", f"{auc:.3f}")
    else:
        st.info("Upload faults CSV (2024) to enable AUROC / confusion matrix.")

# Tab 2: Confusion matrix on FULL dataset (always uses ALL cables; faults only from 2024)
with tab2:
    df_all = st.session_state["df_all"].copy()
    if "ACTUAL_FAIL_24" in df_all.columns:
        df_all = ensure_helper_columns(df_all)

        # Targets & predictions (Poor/Moderate → 1, Good → 0), use health_band from CSV
        y_true = df_all["ACTUAL_FAIL_24"].astype(int)
        y_pred_bin = df_all["health_band"].map({"Poor":1, "Moderate":1, "Good":0}).astype(int)

        cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred OK","Pred Fail"],
                    yticklabels=["Actual OK","Actual Fail"], ax=ax)
        ax.set_xlabel("Prediction"); ax.set_ylabel("Reality")
        st.pyplot(fig, use_container_width=False)

        # Metrics on full set
        acc  = (tp + tn) / cm.sum()
        prec = precision_score(y_true, y_pred_bin, zero_division=0)
        rec  = recall_score   (y_true, y_pred_bin, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) else 0
        f1   = f1_score       (y_true, y_pred_bin, zero_division=0)
        auc  = roc_auc_score  (y_true, 1 - df_all["health_score"]/100)

        auc_tag = ("chance-level" if auc < .6 else "fair" if auc < .7
                   else "good" if auc < .8 else "excellent")

        st.markdown(f"""
**TP** = {tp}  | **FP** = {fp}  | **FN** = {fn}  | **TN** = {tn}

| Metric | Value |
|--------|-------|
| **Accuracy**    | `{acc:.2%}` |
| **Precision**   | `{prec:.2%}` |
| **Recall**      | `{rec:.2%}` |
| **Specificity** | `{spec:.2%}` |
| **F1-score**    | `{f1:.2%}` |
| **AUROC**       | `{auc:.3f}` ({auc_tag}) |
""")
    else:
        st.info("Confusion matrix unavailable — upload a faults CSV so we can build ACTUAL_FAIL_24 (from 2024 only).")

# Tab 3: Data table (filtered)
with tab3:
    cols = ["SWNO","SOURCE_SS","DESTINATION_SS","health_score_10",
            "health_band","failure_prob","primary_health_driver","pred_faults_2024"]
    cols = [c for c in cols if c in df_f.columns]
    st.dataframe(df_f.sort_values("health_score_10", ascending=True)[cols])

# ═════════════════════════════════════════════════╗
# 6. legends (discrete, high contrast)            ║
# ═════════════════════════════════════════════════╝
st.markdown("### Legends")

l1, l2 = st.columns(2)
with l1:
    st.markdown("**Health Band (from CSV)**")
    band_labels = [("Good", BAND_COLORS.get("Good","#228B22")),
                   ("Moderate", BAND_COLORS.get("Moderate","#FF8C00")),
                   ("Poor", BAND_COLORS.get("Poor","#B22222"))]
    fig, ax = plt.subplots(figsize=(4, 0.8))
    for idx, (label, col) in enumerate(band_labels):
        ax.bar(idx, 1, color=col)
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels([b for b,_ in band_labels], fontsize=9)
    ax.set_yticks([])
    st.pyplot(fig, use_container_width=True)

with l2:
    st.markdown("**Failure Probability Bands**")
    prob_bands = [
        ("<20%", "#006400"),
        ("20–40%", "#9ACD32"),
        ("40–60%", "#FFD700"),
        ("60–80%", "#FF8C00"),
        (">80%", "#B22222")
    ]
    fig, ax = plt.subplots(figsize=(4, 0.8))
    for idx, (label, col) in enumerate(prob_bands):
        ax.bar(idx, 1, color=col)
    ax.set_xticks(range(len(prob_bands)))
    ax.set_xticklabels([label for label, _ in prob_bands], fontsize=9, rotation=0)
    ax.set_yticks([])
    st.pyplot(fig, use_container_width=True)
