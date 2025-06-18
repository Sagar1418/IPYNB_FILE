import streamlit as st
import pandas as pd
from pyvis.network import Network
import tempfile
import networkx as nx
import random

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

def _extract_cable_type(feederid):
    if feederid is None or pd.isnull(feederid) or feederid == 'nan':
        return "UNKNOWN"
    p = str(feederid).split("_")
    return p[1].upper() if len(p) >= 3 else "UNKNOWN"

def _extract_feeder_token(feederid):
    if feederid is None or pd.isnull(feederid) or feederid == 'nan':
        return "NULL"
    p = str(feederid).split("_")
    return p[2] if len(p) >= 3 else "NULL"

df["CABLE_TYPE"] = df["FEEDERID"].apply(_extract_cable_type)
df["FEEDER_ID"] = df["FEEDERID"].apply(_extract_feeder_token)

# ---- MULTI-SELECT for CABLE_TYPE ----
cable_types = sorted(df["CABLE_TYPE"].dropna().unique())
selected_cable_types = st.multiselect("Select Network Type(s):", cable_types, default=[cable_types[0]])

# ---- FEEDER_ID multi-select for selected cable types ----
# ---- FEEDER_ID multi-select for selected cable types ----
feeders_list = sorted(df[df["CABLE_TYPE"].isin(selected_cable_types)]["FEEDER_ID"].fillna("NULL").unique().tolist())
if "NULL" not in feeders_list:
    feeders_list.append("NULL")

# ADD "All" as first option!
feeder_options = ["All"] + feeders_list
selected_feeders = st.multiselect("Select FEEDER ID(s):", feeder_options, default=["All"])

# Adjust filtering logic: If "All" in selection, use all feeders for selected cable types
if "All" in selected_feeders:
    view_df = df[df["CABLE_TYPE"].isin(selected_cable_types)].copy()
else:
    view_df = df[
        (df["CABLE_TYPE"].isin(selected_cable_types)) &
        (df["FEEDER_ID"].isin(selected_feeders))
    ].copy()

view_df["SOURCE_SS"] = view_df["SOURCE_SS"].fillna("UNKNOWN").astype(str)
view_df["DESTINATION_SS"] = view_df["DESTINATION_SS"].fillna("UNKNOWN").astype(str)

MAX_EDGES = 500
MAX_EDGES = st.number_input(
    "Maximum number of edges to show (for performance):",
    min_value=1, max_value=50000, value=MAX_EDGES, step=1000,
    help="Limits the number of edges shown in the network visualization for performance reasons."
)
if len(view_df) > MAX_EDGES:
    st.warning(f"Network has {len(view_df)} edges! Only first {MAX_EDGES} shown for clarity.")
    view_df = view_df.head(MAX_EDGES)

pair_counts = view_df.groupby(["SOURCE_SS", "DESTINATION_SS"]).size()
multi_edge_pairs = pair_counts[pair_counts > 1].index.tolist()

def random_color():
    return "#"+''.join(random.choices('0123456789ABCDEF', k=6))

edges = []
color_map = {}
for idx, row in view_df.iterrows():
    src = row["SOURCE_SS"]
    dst = row["DESTINATION_SS"]
    is_multi = (src, dst) in multi_edge_pairs
    color = ""
    # For unique color even if row content same (so table, edge always match)
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
# ---- EDGE TYPE FILTER ----
edge_type = st.selectbox(
    "Select which type of edges to show:",
    ["All", "Self-loop only", "Multiple edges only", "Single edges only"],
    index=0
)

# Identify all self-loops, multi, single
pair_counts = view_df.groupby(["SOURCE_SS", "DESTINATION_SS"]).size()
multi_edge_pairs = pair_counts[pair_counts > 1].index.tolist()
single_edge_pairs = pair_counts[pair_counts == 1].index.tolist()

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

# Apply edge filter to the DataFrame
view_df = view_df[view_df.apply(edge_filter, axis=1)].copy()


G = nx.MultiDiGraph()
for e in edges:
    G.add_node(e["src"])
    G.add_node(e["dst"])
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
st.subheader(f"Explanation for Multiple Edges Only (Network Type(s): {', '.join(selected_cable_types)} | FEEDER(s): {', '.join(selected_feeders)})")

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
