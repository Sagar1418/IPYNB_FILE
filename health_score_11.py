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

st.set_page_config("11 kV Feeder Health (FAST)", layout="wide")
st.title("11 kV Feeder / Cable Health — Advanced Chain View")

uploaded = st.file_uploader("Upload **AFINAL_full.csv**", type="csv")
if not uploaded:
    st.stop()

# --- Score sharpness gamma slider at TOP ---
gamma = st.sidebar.slider("Score sharpness γ (affects all scores)", 0.01, 3.0, 1.0, 0.1)

# --- Read and clean data ---
df0 = pd.read_csv(uploaded, low_memory=False)
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
    "CBL_FAULT_COUNT":         0.18,
    "CBL_MAX_REPAIR_HRS":      0.08,
    "CBL_AVG_REPAIR_HRS":      0.04,
    "AGE":                     0.10,
    "SWITCH_LOAD_FACTOR_MEAN": 0.08,
    "SWITCH_Y_INST_VOLTAGE_STD": 0.08,
    "LENGTH":                  0.05,
    "NOOFJOINTS":              0.07,
    "SECTIONLOSS_KW":          0.08,
    "DESIGN_RISK":             0.15,
    "RESISTANCE":              0.04,
    "FEEDER_LOSS_FACTOR_MEAN": 0.05,
}

year = datetime.now().year
df0["AGE"] = year - pd.to_datetime(df0["DATECREATED"], errors="coerce").dt.year.fillna(year)

def design_risk(csize, ctype):
    s = (str(csize) + str(ctype)).upper()
    if "PILC" in s: return 1.0
    if " AL" in s: return 0.8
    if any(x in s for x in ("XLPE","CU","COPPER")): return 0.2
    return 0.5

df0["DESIGN_RISK"] = [design_risk(a, b) for a, b in zip(df0.get("CABLESIZE", ""), df0.get("CLUSTER_TYPE", ""))]

normed = pd.DataFrame(index=df0.index)
for col in weights:
    vals = pd.to_numeric(df0.get(col, np.nan), errors="coerce")
    vals_clip = mstats.winsorize(vals, limits=[0.01, 0.01])
    median = np.nanmedian(vals_clip)
    vals_filled = np.where(np.isnan(vals_clip), median, vals_clip)
    minv, maxv = np.nanmin(vals_filled), np.nanmax(vals_filled)
    normed[col] = (vals_filled - minv) / (maxv - minv + 1e-8)
    if minv == maxv:
        normed[col] = 0

risk = np.zeros(len(df0))
for col, w in weights.items():
    risk += w * normed[col]
risk_scaled = (risk - risk.min()) / (risk.max() - risk.min() + 1e-8)
risk_final = np.power(risk_scaled, gamma).clip(0, 1)
df0["CABLE_HEALTH"] = (1 + 9 * (1 - risk_final)).round().astype("Int16")

# --- SHOW OVERALL HEALTH HISTOGRAM (this will update with gamma) ---
hist_all = alt.Chart(df0).mark_bar(color="#047ecf").encode(
    x=alt.X("CABLE_HEALTH:Q", bin=True, title="Health (1 worst → 10 best)"),
    y=alt.Y("count()", title="Cable Count")
)
st.altair_chart(hist_all, use_container_width=True)

# --- Filtering ---
feeders = sorted(df0["FEEDER_ID"].dropna().astype(int).unique())
feeder_options = ["All"] + [str(f) for f in feeders]
sel = st.sidebar.multiselect("Select FEEDER_ID(s):", feeder_options, default=["All"])
if "All" in sel or not sel:
    sel_feeders = set(feeders)
else:
    sel_feeders = set(map(int, sel))

edge_cap = st.sidebar.slider("Max cables (lowest health first)", 1000, 50000, 500, 100)

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
    chain = vis[vis["FEEDER_ID"]==fid].sort_values("RANK")
    chains[fid] = [(str(row[src_col]), str(row[dst_col]), row) for _, row in chain.iterrows()]

# --- Build the network graph ---
cmap = matplotlib.cm.get_cmap("RdYlGn")
def color(score:int)->str:
    r,g,b = cmap((score-1)/9)[:3]
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

worst, node_row, G = {}, {}, nx.MultiDiGraph()
for fid, chain in chains.items():
    for s, d, row in chain:
        hc = int(row["CABLE_HEALTH"])
        for node in (s, d):
            worst[node] = min(worst.get(node,10), hc)
            node_row[node] = node_row.get(node, []) + [row]
for fid, chain in chains.items():
    for i, (s, d, row) in enumerate(chain):
        hc = int(row["CABLE_HEALTH"])
        cable_info = (
            f"FROM_SWITCH: {row.get('FROM_SWITCH','-')}\n"
            f"TO_SWITCH: {row.get('TO_SWITCH','-')}\n"
            f"CLUSTER_TYPE: {row.get('CLUSTER_TYPE','-')}\n"
            f"Health: {hc}/10\n"
            f"Faults: {row.get('CBL_FAULT_COUNT','-')}\n"
            f"Length: {row.get('LENGTH','-')}\n"
            f"COMMENTS: {row.get('COMMENTS','-')}\n"
        )
        for node in (s, d):
            G.add_node(str(node), color=color(worst[node]), size=30 + 2*(10-worst[node]),
                title=f"{node}\nWorst cable health: {worst[node]}/10\nCables: {len(node_row[node])}")
        G.add_edge(str(s), str(d), label=f"{hc}", color=color(hc),
                   width=3 + hc/2, title=cable_info)

if len(vis) > 0:
    net = Network(height="1000px", width="100%", directed=True, bgcolor="#fff")
    net.from_nx(G)
    net.set_options(json.dumps({
        "interaction": {"hover": True, "multiselect": True},
        "edges": {"arrows": {"to": {"enabled": True}}, "smooth": {"type": "dynamic"}},
        "physics": {
            "enabled": True,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -80},
            "stabilization": {"enabled": True, "iterations": 150, "fit": True}
        }
    }))
    net.html += """
    <script>
    setTimeout(function(){
      try { if(window.network){network.setOptions({physics:false});} }catch(e){}
    }, 3000);
    </script>
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.save_graph(tmp.name); tmp.seek(0)
        st.subheader("Cable Chain Network (selected feeder(s), static after 3 sec)")
        st.components.v1.html(tmp.read().decode(), height=1000, scrolling=True)

    # Health histogram for shown cables
    st.subheader("Cable Health Distribution (selected cables, current γ)")
    hist = alt.Chart(vis).mark_bar(color="#047ecf").encode(
        x=alt.X("CABLE_HEALTH:Q", bin=True, title="Health (1 worst → 10 best)"),
        y=alt.Y("count()", title="Cable Count")
    )
    st.altair_chart(hist, use_container_width=True)

    # Show cable table
    st.subheader("Essential Cable Table")
    ess = ["FEEDER_ID","FROM_SWITCH","TO_SWITCH",src_col,dst_col,"LENGTH","CBL_FAULT_COUNT","CLUSTER_TYPE","CABLE_HEALTH","COMMENTS"]
    st.dataframe(vis[ess].sort_values("CABLE_HEALTH"))

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

st.info(
    "• Cable health scores and all histograms update *immediately* to your gamma value, for all cables.\n"
    "• Select feeder(s) to see the network graph for those only.\n"
    "• Hover or click on any cable or node in the network to view all details in the tooltip popup.\n"
)
