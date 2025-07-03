import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import altair as alt
import matplotlib
import json
import tempfile
from datetime import datetime

st.set_page_config("11 kV Feeder Health (FAST)", layout="wide")
st.title("11 kV Feeder / Cable Health — Advanced Chain View")

uploaded = st.file_uploader("Upload **AFINAL_full.csv**", type="csv")
if not uploaded:
    st.stop()

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
feeders = sorted(df0["FEEDER_ID"].dropna().astype(int).unique())
feeder_options = ["All"] + [str(f) for f in feeders]
sel = st.sidebar.multiselect("Select FEEDER_ID(s):", feeder_options, default=["All"])
if "All" in sel or not sel:
    sel_feeders = set(feeders)
else:
    sel_feeders = set(map(int, sel))

edge_cap = st.sidebar.slider("Max cables (lowest health first)", 100, 5000, 700, 100)
gamma    = st.sidebar.slider("Score sharpness γ", 0.5, 3.0, 1.0, 0.1)

df = df0[df0["FEEDER_ID"].isin(sel_feeders)].copy()
if df.empty:
    st.warning("No cables for selected feeders.")
    st.stop()

# --- Health scoring factors (real-life advanced weights) ---
weights = {
    "CBL_FAULT_COUNT":         0.25,
    "CBL_MAX_REPAIR_HRS":      0.10,
    "AGE":                     0.12,
    "SWITCH_LOAD_FACTOR_MEAN": 0.08,
    "SWITCH_Y_INST_VOLTAGE_STD": 0.06,
    "LENGTH":                  0.07,
    "NOOFJOINTS":              0.10,
    "SECTIONLOSS_KW":          0.06,
    "DESIGN_RISK":             0.16,
}

year = datetime.now().year
df["AGE"] = year - pd.to_datetime(df.get("DATECREATED"), errors="coerce").dt.year.fillna(year)
def design_risk(csize, ctype):
    s = (str(csize) + str(ctype)).upper()
    if "PILC" in s: return 1.0
    if " AL" in s: return 0.8
    if any(x in s for x in ("XLPE","CU","COPPER")): return 0.2
    return 0.5
df["DESIGN_RISK"] = [design_risk(a, b) for a, b in zip(df.get("CABLESIZE",""), df.get("CLUSTER_TYPE",""))]

for col in weights:
    df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(float)

mat = df[list(weights)].astype(float)
norm = (mat - mat.min()) / (mat.max() - mat.min()).replace(0,1)
risk = np.zeros(len(norm))
for col, w in weights.items():
    risk += w * norm[col]
risk = (risk - risk.min()) / (risk.max() - risk.min() + 1e-6)
risk = np.power(risk, gamma).clip(0, 1)
df["CABLE_HEALTH"] = (1 + 9*(1-risk)).round().astype("Int16")

contr_mat = norm * np.array([w for w in weights.values()])
contr_mat = pd.DataFrame(contr_mat, columns=list(weights))
def top3_contrib(row):
    arr = row.astype(float)
    return arr.abs().nlargest(3)
top_contribs = contr_mat.apply(top3_contrib, axis=1)

# --- Display only lowest-health cables first ---
vis = df.sort_values("CABLE_HEALTH").head(edge_cap)
chains = {}
for fid in vis["FEEDER_ID"].unique():
    chain = vis[vis["FEEDER_ID"]==fid].sort_values("RANK")
    chains[fid] = [(str(row[src_col]), str(row[dst_col]), row) for _, row in chain.iterrows()]

# --- Build the network graph (no isolates!) ---
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
        for node in (s, d):
            G.add_node(str(node), color=color(worst[node]), size=30 + 2*(10-worst[node]),
                title=f"{node}\nWorst cable health: {worst[node]}/10\nCables: {len(node_row[node])}")
        idx = row.name
        top3 = top_contribs.loc[idx]
        txt = "\n".join(f"{k}: {top3[k]:.2f}" for k in top3.index)
        cable_info = (
            f"FROM_SWITCH: {row.get('FROM_SWITCH','-')}\n"
            f"TO_SWITCH: {row.get('TO_SWITCH','-')}\n"
            f"CLUSTER_TYPE: {row.get('CLUSTER_TYPE','-')}\n"
            f"Health: {hc}/10\n"
            f"Faults: {row.get('CBL_FAULT_COUNT','-')}\n"
            f"Length: {row.get('LENGTH','-')}\n"
            f"COMMENTS: {row.get('COMMENTS','-')}\n"
            f"Top Contributors:\n{txt}"
        )
        G.add_edge(str(s), str(d), label=f"{hc}", color=color(hc),
                   width=3 + hc/2, title=cable_info)

net = Network(height="700px", width="100%", directed=True, bgcolor="#fff")
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
    st.subheader("Cable Chain Network (Advanced/FAST, static after 3 sec)")
    st.components.v1.html(tmp.read().decode(), height=720, scrolling=True)

# --- Health histogram for shown cables ---
st.subheader("Cable Health Distribution (shown network only)")
hist = alt.Chart(vis).mark_bar(color="#e31a1c").encode(
    x=alt.X("CABLE_HEALTH:Q", bin=True, title="Health (1 worst → 10 best)"),
    y=alt.Y("count()", title="Cable Count")
)
st.altair_chart(hist, use_container_width=True)

# --- Show cable table ---
st.subheader("Essential Cable Table")
ess = ["FEEDER_ID","FROM_SWITCH","TO_SWITCH",src_col,dst_col,"LENGTH","CBL_FAULT_COUNT","CLUSTER_TYPE","CABLE_HEALTH","COMMENTS"]
st.dataframe(vis[ess].sort_values("CABLE_HEALTH"))

st.info("• Hover or click on any cable or node in the network to view all details in the tooltip popup. "
  )

