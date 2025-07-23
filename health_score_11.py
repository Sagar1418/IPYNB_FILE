import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
import json
import tempfile
from datetime import datetime
from scipy.stats import mstats
import textwrap

st.set_page_config("11 kV Feeder Health (FAST)", layout="wide")
st.title("11 kV Feeder / Cable Health — Chain View")

uploaded = st.file_uploader("Upload **AFINAL_full.csv**", type="csv")
if not uploaded:
    st.stop()

# --- Score sharpness gamma slider at TOP ---
# gamma = st.sidebar.slider("Score sharpness γ (affects all scores)", 0.01, 3.0, 0.5, 0.1)
gamma = 0.45 # Default value for testing, can be changed via slider

# --- Read and clean data ---
df0 = pd.read_csv(uploaded, low_memory=False)
df0 = df0[~df0['REMARKS'].str.upper().isin(['ABANDONED', 'DISCONNECTED'])]
df0.columns = [c.upper() for c in df0.columns]
if "COMMENTS" in df0.columns:
    df0["COMMENTS"] = df0["COMMENTS"].astype(str).str.replace(r"<br\s*/?>", "\n", regex=True)

def pick(cols, candidates):
    for name in candidates:
        if name in cols:
            return name
    return ""

src_col = pick(df0.columns, ["SOURCE_SS","SOURCE_LOCATION","SRC_STATION"])
dst_col = pick(df0.columns, ["DESTINATION_SS","DESTINATION_LOCATION","DST_STATION"])
if not src_col: src_col = dst_col
if not dst_col: dst_col = src_col
if not src_col or not dst_col:
    st.error("Could not detect source/dest station columns.")
    st.stop()

df0["FEEDER_ID"] = pd.to_numeric(df0["FEEDER_ID"], errors="coerce").astype("Int64")

# --- Calculate health score ONCE for all cables (ALWAYS with selected gamma) ---
weights = {
    "FAULT_SWITCH_COUNT":         0.25,
    "CBL_MAX_REPAIR_HRS":      0.16,
    "CBL_AVG_REPAIR_HRS":      0.12,
    "AGE":                     0.12,
    "AGG_MEASUREDLENGTH":        0.15,
    
    "DESIGN_RISK":             0.10, 
    "NO_OF_SEGMENT":           0.10,
    
}

year = datetime.now().year
df0["AGE"] = year - pd.to_datetime(df0["DATECREATED"], errors="coerce").dt.year.fillna(year)

def design_risk(csize, ctype):
    
    s = (str(csize) + str(ctype)).upper()
    if "PILC" in s: return 1.0
    if " AL" in s: return 0.8
    if any(x in s for x in ("XLPE","CU","COPPER")): return 0.2
    return 0.5

df0["DESIGN_RISK"] = [design_risk(a, b) for a, b in zip(df0.get("CABLETYPE", ""), df0.get("CABLECONDUCTORMATERIAL", ""))]

normed = pd.DataFrame(index=df0.index)
for col in weights:
    
    if col not in df0.columns:
        vals = pd.Series([np.nan] * len(df0), index=df0.index)
    else:
        vals = pd.to_numeric(df0[col], errors="coerce")
    vals_clip = mstats.winsorize(vals, limits=[0.01, 0.01])
    median = np.nanmedian(vals_clip)
    vals_filled = np.where(np.isnan(vals_clip), median, vals_clip)
    minv, maxv = np.nanmin(vals_filled), np.nanmax(vals_filled)
    normed[col] = (vals_filled - minv) / (maxv - minv + 1e-8)
    if minv == maxv:
        normed[col] = 0

def sigmoid(x, k=5):
    return 1 / (1 + np.exp(-k * (x - 0.5)))

risk = np.zeros(len(df0))
for col, w in weights.items():
    sig = sigmoid(normed[col].fillna(0), k=5)  # You can tune 'k'
    risk += w * sig

# Normalize the final risk to [0, 1] range
risk_scaled = (risk - risk.min()) / (risk.max() - risk.min() + 1e-8)
risk_final = np.power(risk_scaled, gamma).clip(0, 1)
# Identify top 3 contributing features per row
top3_features = []
for i in normed.index:
    contribs = {}
    for col, w in weights.items():
        val = sigmoid(normed.at[i, col], k=5) if col in normed.columns else 0
        contribs[col] = val * w
    top = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_features.append(", ".join([f for f, _ in top]))

df0["TOP3_CONTRIBUTORS"] = top3_features

df0["CABLE_HEALTH"] = (1 + 9 * (1 - risk_final)).round().astype("Int16")

# --- SHOW OVERALL HEALTH HISTOGRAM (this will update with gamma) ---
# hist_all = alt.Chart(df0).mark_bar(color="#047ecf").encode(
#     x=alt.X("CABLE_HEALTH:Q", bin=True, title="Health (1 worst → 10 best)"),
#     y=alt.Y("count()", title="Cable Count")
# )
hist = alt.Chart(df0).mark_bar(color="#047ecf").encode(
    x=alt.X("CABLE_HEALTH:O", title="Health Score (1 worst → 10 best)"),
    y=alt.Y("count()", title="Cable Count")
)

st.altair_chart(hist, use_container_width=True)
# st.altair_chart(hist_all, use_container_width=True)

# --- Filtering ---
feeders = sorted(df0["FEEDER_ID"].dropna().astype(int).unique())
feeder_options = ["All"] + [str(f) for f in feeders]
sel = st.sidebar.multiselect("Select FEEDER_ID(s):", feeder_options, default=["All"])
if "All" in sel or not sel:
    sel_feeders = set(feeders)
else:
    sel_feeders = set(map(int, sel))

edge_cap = st.sidebar.slider("Max cables (lowest health first)", 1000, 50000, 500, 100)
# ── Edge‑label toggle ─────────────────────────────────────────
show_rank = st.sidebar.checkbox(
    "Show RANK on edges (instead of Health Score)",
    value=False
)

# ---- FILTER ONLY AFTER health is calculated -----
df = df0[df0["FEEDER_ID"].isin(sel_feeders)].copy()
if df.empty:
    st.warning("No cables for selected feeders.")
    st.stop()

# --- Remove NaN/empty stations (do not show those nodes) ---
def safe_str(x):
    return str(x) if pd.notna(x) and str(x).strip() and str(x).upper() not in {"NAN", "NONE", "-"} else None
df[src_col] = df[src_col].map(safe_str)
df[dst_col] = df[dst_col].map(safe_str)
df = df[df[src_col].notna() & df[dst_col].notna()]

vis = df.sort_values("CABLE_HEALTH").head(edge_cap)
chains = {}
for fid in vis["FEEDER_ID"].unique():
    chain = vis[vis["FEEDER_ID"]==fid]
    chains[fid] = [(str(row[src_col]), str(row[dst_col]), row) for _, row in chain.iterrows()]

# --- Build the network graph ---
cmap = matplotlib.cm.get_cmap("RdYlGn")
def color(score:int)->str:
    r,g,b = cmap((score-1)/9)[:3]
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

worst, node_row, G = {}, {}, nx.MultiDiGraph()

for fid, chain in chains.items():
    for i, (s, d, row) in enumerate(chain):
        hc   = int(row["CABLE_HEALTH"])
        rank = row.get("RANK", "")          # assume your CSV has a RANK column

        # choose which label to draw
        label_value = str(rank) if show_rank else str(hc)

        # (optional) include RANK in the tooltip as well:
        cable_info = (
            f"Rank: {rank}\n"
            f"FROM_SWITCH: {row.get('FROM_SWITCH','-')}\n"
            f"TO_SWITCH:   {row.get('TO_SWITCH','-')}\n"
            f"Health:     {hc}/10\n"
            f"Top 3 Contributors: {row.get('TOP3_CONTRIBUTORS','-')}\n"
            f"Faults:     {row.get('FAULT_SWITCH_COUNT','-')}\n"
            f"Length:     {row.get('LENGTH','-')}\n"
            f"Remarks:    {row.get('REMARKS','-')}\n"
            f"\nPath: {textwrap.fill(str(row.get('PATH','-')),60)}"
        )

        # add nodes (unchanged)
        for node in (s, d):
            worst[node]  = min(worst.get(node,10), hc)
            node_row[node] = node_row.get(node,[]) + [row]

        # add the edge with our dynamic label
        G.add_edge(
            str(s), str(d),
            label = label_value,
            color = color(hc),
            width = 3 + hc/2,
            title = cable_info
        )

if len(vis) > 0:
    net = Network(height="2000px", width="100%", directed=True, bgcolor="#fff")
    net.from_nx(G)
    net.set_options("""
    var options = {
    "nodes": {"size": 40},
    "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
        "smooth": {"type": "dynamic"}
    },
    "physics": {
        "forceAtlas2Based": {
        "gravitationalConstant": -80,
        "centralGravity": 0.005,
        "springLength": 180,
        "springConstant": 0.08
        },
        "maxVelocity": 200,
        "solver": "forceAtlas2Based",
        "timestep": 0.60,
        "stabilization": {"enabled": true, "iterations": 20 }
    }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
        net.save_graph(temp_file.name)
        temp_file.seek(0)
        html_content = temp_file.read().decode()

        # Inject script to stop physics after a short delay
        html_content += """
        <script type="text/javascript">
        setTimeout(function() {
            if (window.network) {
            network.setOptions({physics: false});
            }
        }, 30000);
        </script>
        """

    # st.components.v1.html(html_content, height=810, scrolling=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name); tmp.seek(0)
        st.subheader("Cable Chain Network Graph")
        # st.components.v1.html(tmp.read().decode(), height=1000, scrolling=True)
        st.components.v1.html(html_content, height=2000, scrolling=True)

    # Health histogram for shown cables
    st.subheader("Cable Health Distribution (selected FEEDER_IDs)")
    hist2 = alt.Chart(vis).mark_bar(color="#047ecf").encode(
    x=alt.X("CABLE_HEALTH:O", title="Health Score (1 worst → 10 best)"),
    y=alt.Y("count()", title="Cable Count")
)
    # hist = alt.Chart(vis).mark_bar(color="#047ecf").encode(
    #     x=alt.X("CABLE_HEALTH:Q", bin=True, title="Health (1 worst → 10 best)"),
    #     y=alt.Y("count()", title="Cable Count")
    # )
    st.altair_chart(hist2, use_container_width=True)

    # Show cable table
    # st.subheader("Essential Cable Table")
    # ess = ["FEEDER_ID","FROM_SWITCH","TO_SWITCH",src_col,dst_col,"LENGTH","CBL_FAULT_COUNT","CLUSTER_TYPE","CABLE_HEALTH","COMMENTS"]
    # st.dataframe(vis[ess].sort_values("CABLE_HEALTH"))

# --- Color map legend ---
st.markdown("#### Health Score Color Map")
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cmap = matplotlib.cm.RdYlGn
norm = matplotlib.colors.Normalize(vmin=1, vmax=10)
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap,
                            norm=norm,
                            orientation='horizontal')
cb1.set_label('Cable Health (1 = worst, 10 = best)')
st.pyplot(fig)
# --- Show substation names ---
st.markdown("#### Substation Names")
substations = sorted(set(df[src_col].dropna().unique()).union(set(df[dst_col].dropna().unique())))
if substations:
    st.write("Substation names:", substations)
    st.write("Total substations:", len(substations))
else:
    st.write("No substations found in the data.")
