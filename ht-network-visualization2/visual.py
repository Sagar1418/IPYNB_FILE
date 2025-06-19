import streamlit as st
import pandas as pd
from pyvis.network import Network
import tempfile
import networkx as nx
import random
import re

st.set_page_config(page_title="HT Network Feeder-wise Visualization", layout="wide")
st.title("HT Network: Feeder-wise Visualization & Edge Explanation")

uploaded_file = st.file_uploader("Upload HTCABLE.csv", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.drop_duplicates()
for col in df.columns:
    df[col] = df[col].fillna("-")
df['FEEDERID'] = df['FEEDERID'].astype(str)

# ----- VOLTAGE LEVEL LOGIC -----
def feeder_voltage(feederid):
    if pd.isnull(feederid) or feederid in ["-", "NULL", "nan"]:
        return "Others"
    feederid_upper = str(feederid).upper()
    if "11KV" in feederid_upper:
        return "11kV"
    elif "22KV" in feederid_upper:
        return "22kV"
    elif "33KV" in feederid_upper:
        return "33kV"
    else:
        match = re.search(r'(\d{2})KV', feederid_upper)
        if match:
            return f"{match.group(1)}kV"
        else:
            return "Others"

df["FEEDER_VOLTAGE"] = df["FEEDERID"].apply(feeder_voltage)

def _extract_feeder_token(feederid):
    if feederid is None or pd.isnull(feederid) or feederid == 'nan':
        return "NULL"
    p = str(feederid).split("_")
    return p[2] if len(p) >= 3 else "NULL"

df["FEEDER_ID"] = df["FEEDERID"].apply(_extract_feeder_token)

# ---- VOLTAGE MULTI-SELECT ----
voltage_levels = ["11kV", "22kV", "33kV", "Others"]
selected_voltages = st.multiselect(
    "Select Voltage Level(s):", voltage_levels, default=voltage_levels[:1]
)

# ---- FEEDER_ID multi-select for selected voltage ----
filtered_by_voltage = df[df["FEEDER_VOLTAGE"].isin(selected_voltages)]
feeders_list = sorted(filtered_by_voltage["FEEDER_ID"].fillna("NULL").unique().tolist())
if "NULL" not in feeders_list:
    feeders_list.append("NULL")
feeder_options = ["All"] + feeders_list
selected_feeders = st.multiselect("Select FEEDER ID(s):", feeder_options, default=["All"])

# ---- FINAL FILTER for Data ----
if "All" in selected_feeders:
    view_df = filtered_by_voltage.copy()
else:
    view_df = filtered_by_voltage[filtered_by_voltage["FEEDER_ID"].isin(selected_feeders)].copy()

view_df["SOURCE_SS"] = view_df["SOURCE_SS"].fillna("UNKNOWN").astype(str)
view_df["DESTINATION_SS"] = view_df["DESTINATION_SS"].fillna("UNKNOWN").astype(str)

# ---- EDGE LIMIT ----
MAX_EDGES = 500
MAX_EDGES = st.number_input(
    "Maximum number of edges to show (for performance):",
    min_value=1, max_value=50000, value=MAX_EDGES, step=1000,
    help="Limits the number of edges shown in the network visualization for performance reasons."
)
if len(view_df) > MAX_EDGES:
    st.warning(f"Network has {len(view_df)} edges! Only first {MAX_EDGES} shown for clarity.")
    view_df = view_df.head(MAX_EDGES)

# ---- EDGE TYPE FILTER ----
pair_counts = view_df.groupby(["SOURCE_SS", "DESTINATION_SS"]).size()
multi_edge_pairs = pair_counts[pair_counts > 1].index.tolist()
single_edge_pairs = pair_counts[pair_counts == 1].index.tolist()

edge_type = st.selectbox(
    "Select which type of edges to show:",
    ["All", "Self-loop only", "Multiple edges only", "Single edges only"],
    index=0
)

def edge_filter(row):
    src = row["SOURCE_SS"]
    dst = row["DESTINATION_SS"]
    if edge_type == "All":
        return True
    elif edge_type == "Self-loop only":
        return src == dst
    elif edge_type == "Multiple edges only":
        return (src, dst) in multi_edge_pairs
    elif edge_type == "Single edges only":
        return (src, dst) in single_edge_pairs
    return True

view_df = view_df[view_df.apply(edge_filter, axis=1)].copy()

# ---- Find highlighted nodes: SOURCE_SS where SOURCE_SWITCH_ID == selected FEEDER_ID ----
highlighted_nodes = set()
if "All" not in selected_feeders and "NULL" not in selected_feeders and "SOURCE_SWITCH_ID" in view_df.columns:
    mask = view_df["SOURCE_SWITCH_ID"].isin(selected_feeders)
    highlighted_nodes = set(view_df.loc[mask, "SOURCE_SS"].unique())

# ---- BUILD EDGES ----
def random_color():
    return "#"+''.join(random.choices('0123456789ABCDEF', k=6))

edges = []
color_map = {}
for idx, row in view_df.iterrows():
    src = row["SOURCE_SS"]
    dst = row["DESTINATION_SS"]
    is_multi = (src, dst) in multi_edge_pairs
    color = ""
    row_tuple = tuple(row)
    if is_multi:
        color = color_map.get(row_tuple)
        if not color:
            color = random_color()
            while color in color_map.values():
                color = random_color()
            color_map[row_tuple] = color
    edge_dict = row.to_dict()
    edge_dict.update({"src": src, "dst": dst, "is_multi": is_multi, "color": color})
    edges.append(edge_dict)

# ---- BUILD NETWORKX GRAPH (highlight only matching SOURCE_SS nodes) ----
ss_to_feederid = dict(zip(view_df["SOURCE_SS"], view_df["FEEDER_ID"]))
G = nx.MultiDiGraph()
for e in edges:
    for node in [e["src"], e["dst"]]:
        if node in highlighted_nodes:
            G.add_node(
                node,
                color="#FF3333",
                size=35,
                borderWidth=4,
                label=f"{node} (FEEDER)",
                font={'color': '#111', 'size': 22, 'bold': True},
                title=f"FEEDER_ID: {ss_to_feederid.get(node, '-')}"
            )
        else:
            G.add_node(
                node,
                size=25,
                title=f"FEEDER_ID: {ss_to_feederid.get(node, '-')}"
            )
    edge_kwargs = dict(
        label="",
        title=" | ".join([f"{k}: {e[k]}" for k in ['MEASUREDLENGTH','COMMENTS'] if k in e])
    )
    if e["is_multi"]:
        edge_kwargs["color"] = e["color"]
        edge_kwargs["width"] = 4
    G.add_edge(e["src"], e["dst"], **edge_kwargs)

net = Network(notebook=False, directed=True, width="100%", height="850px", bgcolor="#f4f4f4")
net.from_nx(G)
net.set_options("""
var options = {
  "nodes": {
    "size": 50
  },
  "edges": {
    "arrows": {
      "to": {"enabled": true, "scaleFactor": 0.7}
    },
    "smooth": {"type": "dynamic"}
  },
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -75,
      "centralGravity": 0.01,
      "springLength": 200,
      "springConstant": 0.08
    },
    "maxVelocity": 40,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {"enabled": true, "iterations": 150}
  }
}
""")

for i, e in enumerate(edges):
    if e["is_multi"]:
        net.edges[i]['color'] = e["color"]
        net.edges[i]['width'] = 4
    else:
        net.edges[i]['color'] = "#BBBBBB"
        net.edges[i]['width'] = 1

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
    net.save_graph(temp_file.name)
    st.components.v1.html(temp_file.read().decode(), height=850, scrolling=True)

st.markdown("---")
st.subheader(f"Explanation for Multiple Edges Only (Voltage Level(s): {', '.join(selected_voltages)} | FEEDER(s): {', '.join(selected_feeders)})")

def color_icon_html(color):
    return f"<span style='display:inline-block;width:18px;height:18px;background:{color};border-radius:50%;border:1px solid #222'></span>"

# ---- User-selected columns ----
default_cols = ["Color","SOURCE_SS","DESTINATION_SS","LABELTEXT","MEASUREDLENGTH","REMARK","COMMENTS"]
table_cols = [col for col in default_cols if col in view_df.columns]
all_possible_cols = ["Color"] + [c for c in view_df.columns if c not in ("src","dst","is_multi","color")]

show_cols = st.multiselect(
    "Show columns in explanation table:",
    options=all_possible_cols,
    default=table_cols
)

# ---- Only multi-edge rows, user-selected columns ----
multi_exp_data = []
for e in edges:
    if e["is_multi"]:
        row = {}
        for col in show_cols:
            if col == "Color":
                row[col] = color_icon_html(e["color"])
            else:
                row[col] = e.get(col,"")
        multi_exp_data.append(row)
multi_exp_df = pd.DataFrame(multi_exp_data)

if multi_exp_df.shape[0]:
    st.markdown("**Only those connections are shown below where multiple lines exist between the same Source and Destination.**<br>Row color matches the edge color above.", unsafe_allow_html=True)
    st.write(multi_exp_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.success("No multiple edges found for this selection!")

search = st.text_input("Filter by word in notes (case-insensitive):")
if search:
    mask = view_df['COMMENTS'].str.contains(search, case=False, na=False)
    st.write(f"Showing connections where notes contain: '{search}'")
    st.dataframe(view_df[mask][show_cols])
