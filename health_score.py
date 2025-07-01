import streamlit as st
import pandas as pd
from pyvis.network import Network
import networkx as nx
import matplotlib
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

st.set_page_config(page_title="HT 22kV/33kV Cable Network Health", layout="wide")
st.title("HT 22kV/33kV Cable Network: Unique Cable Health Visualization")

uploaded_file = st.file_uploader("Upload HTCABLE.csv", type=["csv"])
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)
df = df.drop_duplicates()
for col in df.columns:
    df[col] = df[col].fillna("-")

# --- Filter for 22kV/33kV cables only ---
def extract_voltage(col):
    s = str(col).upper()
    if "22KV" in s:
        return "22kV"
    if "33KV" in s:
        return "33kV"
    return "-"
if "VOLTAGE" in df.columns:
    df['VOLTAGE_LEVEL'] = df['VOLTAGE'].apply(extract_voltage)
elif "FEEDERID" in df.columns:
    df['VOLTAGE_LEVEL'] = df['FEEDERID'].apply(extract_voltage)
else:
    st.error("No VOLTAGE or FEEDERID column to detect 22kV/33kV cables!")
    st.stop()
df = df[df['VOLTAGE_LEVEL'].isin(["22kV", "33kV"])]
if df.empty:
    st.warning("No 22kV or 33kV cables found in your file.")
    st.stop()

# --- UI for minimum MEASUREDLENGTH, unique SWNO ---
min_length = st.number_input(
    "Minimum MEASUREDLENGTH to include (units as per your data):",
    min_value=0, max_value=10000, value=50, step=1
)
if 'MEASUREDLENGTH' in df.columns:
    df['MEASUREDLENGTH'] = pd.to_numeric(df['MEASUREDLENGTH'], errors='coerce').fillna(0)
    df = df[df['MEASUREDLENGTH'] > min_length]
    df = df.sort_values('MEASUREDLENGTH', ascending=False)
    df = df.drop_duplicates(subset='SWNO', keep='first')

if df.empty:
    st.warning("No cables remaining after filtering for length and uniqueness.")
    st.stop()

# --- SWNO MULTIselect ---
unique_swnos = sorted(df['SWNO'].unique().tolist())
swno_options = ["All"] + unique_swnos
selected_swnos = st.multiselect(
    "Select SWNO(s) (Cable) to visualize (chain shown if multiple):", 
    swno_options, default=["All"]
)
# If "All" selected (or none), show all. Else, filter to only selected.
if "All" not in selected_swnos:
    df = df[df['SWNO'].isin(selected_swnos)]
    if df.empty:
        st.warning("No cable with these SWNOs found after filtering.")
        st.stop()

# --- Advanced Health Score function (Num_Faults = 50%) ---
def advanced_health_score(df):
    # --- Domain weights (edit as needed, Num_Faults=0.30) ---
    weights = {
        'Num_Faults': 0.30,
        'CURRENT_MEAN': 0.10,
        'CURRENT_STD': 0.08,
        'CABLE_AGE': 0.07,
        'MEASUREDLENGTH': 0.05,
        'Avg_Load_MEAN': 0.08,
        'Avg_Load_STD': 0.04,
        'PEAK_MEAN': 0.06,
        'PEAK_STD': 0.04,
        'CYCLE_MEAN': 0.05,
        'CYCLE_STD': 0.03,
        'OVR_MEAN': 0.04,
        'OVR_STD': 0.03,
        'FUSE_COUNT': 0.02,
        'CABLETYPE_RISK': 0.005,
        'MATERIAL_RISK': 0.005
    }

    # --- Feature Engineering ---
    now_year = datetime.now().year
    if 'DATECREATED' in df.columns:
        years = pd.to_datetime(df['DATECREATED'], errors='coerce').dt.year
        df['CABLE_AGE'] = now_year - years.fillna(now_year)

    # Currents
    current_cols = [c for c in df.columns if c.startswith('I_Month_')]
    if current_cols:
        currents = df[current_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df['CURRENT_MEAN'] = currents.mean(axis=1)
        df['CURRENT_STD'] = currents.std(axis=1)

    # Avg_Load, PEAK, CYCLE, OVR (MEAN/STD)
    for prefix, name in [('Avg_Load_Month_', 'Avg_Load'), 
                         ('PEAK_Month_', 'PEAK'), 
                         ('CYCLE_Month_', 'CYCLE'), 
                         ('OVR_Month_', 'OVR')]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            arr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            df[f'{name}_MEAN'] = arr.mean(axis=1)
            df[f'{name}_STD'] = arr.std(axis=1)

    # Cable/material risk
    if 'CABLETYPE' in df.columns:
        df['CABLETYPE_RISK'] = df['CABLETYPE'].apply(lambda t: 1 if 'PILC' in str(t).upper() else (0.2 if 'XLPE' in str(t).upper() else 0.5))
    if 'CABLECONDUCTORMATERIAL' in df.columns:
        df['MATERIAL_RISK'] = df['CABLECONDUCTORMATERIAL'].apply(lambda m: 0.7 if 'AL' in str(m).upper() else (0.2 if 'CU' in str(m).upper() or 'COPPER' in str(m).upper() else 0.5))
    # FUSE_COUNT
    fuse_cols = [col for col in df.columns if col.startswith('FAULT_RELAY_FUSE')]
    if fuse_cols:
        df['FUSE_COUNT'] = df[fuse_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1)

    # --- SCORING ONLY ON AVAILABLE FEATURES ---
    used = {k: v for k, v in weights.items() if k in df.columns}
    total_weight = sum(used.values())
    if total_weight == 0:
        raise ValueError("No scoring features found in this data!")

    # Compute normalized weighted risk
    score = pd.Series(0, index=df.index, dtype=float)
    for k, w in used.items():
        vals = pd.to_numeric(df[k], errors='coerce').fillna(0)
        rng = vals.max() - vals.min()
        if rng == 0:
            continue
        normed = (vals - vals.min()) / rng
        score += normed * (w / total_weight)
    
    # Health: 1 (worst) to 10 (best)
    df['Health_Score_Advanced'] = (1 + 9 * (1 - score)).round().astype(int)
    df.attrs['health_score_features'] = {k: w/total_weight for k, w in used.items()}
    return df

df = advanced_health_score(df)
df['Health_Score'] = df['Health_Score_Advanced']

# --- Show features and weights used ---
if hasattr(df, "attrs") and "health_score_features" in df.attrs:
    st.markdown("#### Health Score Feature Weights Used (this data):")
    for col, eff_weight in df.attrs['health_score_features'].items():
        st.write(f"{col}: {eff_weight:.2%}")

# --- Node Health (worst cable health attached) ---
node_health = {}
for idx, row in df.iterrows():
    src = row['SOURCE_SS']
    dst = row['DESTINATION_SS']
    score = int(row['Health_Score'])
    for n in [src, dst]:
        if n not in node_health:
            node_health[n] = score
        else:
            node_health[n] = min(node_health[n], score)  # node health = worst attached cable

# --- Network Visualization ---
def health_color(score):
    cmap = matplotlib.cm.get_cmap('RdYlGn')
    rgb = cmap((score-1)/9)[:3]
    return '#{:02x}{:02x}{:02x}'.format(*(int(255*x) for x in rgb))

G = nx.MultiDiGraph()
for idx, row in df.iterrows():
    src = row["SOURCE_SS"]
    dst = row["DESTINATION_SS"]
    swno = row['SWNO'] if 'SWNO' in row else f"edge_{idx}"
    health_score = int(row['Health_Score'])
    edge_color = health_color(health_score)
    label = f"Health: {health_score}/10"
    faults = f"Num_Faults: {row.get('Num_Faults','-')}"
    comments = row.get('COMMENTS','-')
    length = row.get('MEASUREDLENGTH', '-')
    # --------- EDGE TOOLTIP FIXED ----------
    tooltip = (
        f"SWNO: {swno}\n"
        f"Health: {health_score}/10\n"
        f"{faults}\n"
        f"Comments: {comments}\n"
        f"Length: {length}"
    )
    # --------- NODE TOOLTIP FIXED ----------
    for n in [src, dst]:
        nh = node_health[n]
        ncolor = health_color(nh)
        connected_swnos = df[(df['SOURCE_SS'] == n) | (df['DESTINATION_SS'] == n)]['SWNO'].unique()
        connected_swnos_str = ', '.join(str(s) for s in connected_swnos)
        node_tooltip = (
            f"{n}\n"
            f"Node health (worst): {nh}/10\n"
            f"Connected SWNO(s): {connected_swnos_str}"
        )
        G.add_node(
            n,
            title=node_tooltip,
            color=ncolor,
            size=30 + 2*(10-nh)
        )
    G.add_edge(src, dst, color=edge_color, label=label, title=tooltip, width=3+health_score/2)

net = Network(notebook=False, directed=True, width="100%", height="800px", bgcolor="#fff")
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
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
    net.save_graph(temp_file.name)
    temp_file.seek(0)
    html_content = temp_file.read().decode()
    st.components.v1.html(html_content, height=810, scrolling=True)

# --- Summary Visuals ---
st.markdown("### Health Score Distribution")
st.bar_chart(df['Health_Score'].value_counts().sort_index())

st.markdown("### Top 10 Unhealthy Cables (Lowest Health Score)")
top_bad = df.sort_values('Health_Score').head(10)
showcols = ['SWNO','SOURCE_SS','DESTINATION_SS','Health_Score','Num_Faults','COMMENTS','MEASUREDLENGTH']
# Add advanced features if present
showcols += [col for col in ['CURRENT_MEAN','CURRENT_STD','CABLE_AGE','node_connectivity','affected_station_count'] if col in df.columns]
showcols = [c for c in showcols if c in df.columns]
st.dataframe(top_bad[showcols])

st.markdown("### All Unique Cable Health Table")
st.dataframe(df[showcols].sort_values('Health_Score'))

# --- Color Legend ---
st.markdown("#### Health Score Color Legend (1=Red, 10=Green)")
fig, ax = plt.subplots(figsize=(5, 0.4))
cmap = matplotlib.cm.get_cmap('RdYlGn')
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect='auto', cmap=cmap)
ax.set_axis_off()
ax.set_title("1 (Worst)      →      10 (Best)", fontsize=12)
st.pyplot(fig, use_container_width=True)

st.info("Edge/node color: Green = Best health, Red = Worst. Hover for info. Node color shows worst cable health attached to that substation. Selecting multiple SWNOs builds the chain across those cables (by SOURCE_SS/DESTINATION_SS).")
