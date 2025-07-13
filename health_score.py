import streamlit as st
import pandas as pd
from pyvis.network import Network
import networkx as nx
import matplotlib
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import textwrap

st.set_page_config(page_title="HT 22kV/33kV Cable Network Health", layout="wide")
st.title("HT 22kV/33kV Cable Network: Unique Cable Health Visualization")

# --- MODE SELECTION FIRST ---
viz_mode = st.radio(
    "Choose visualization type (choose before upload):",
    ("With Comments", "With Path")
)

uploaded_file = st.file_uploader("Upload SWNO_MASTER_COMBINED_FULL.csv", type=["csv"])
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

# --- Filter by measured length ---
min_length = st.number_input("Minimum MEASUREDLENGTH to include:", min_value=0, max_value=10000, value=200, step=10)
if 'MEASUREDLENGTH' in df.columns:
    df['MEASUREDLENGTH'] = pd.to_numeric(df['MEASUREDLENGTH'], errors='coerce').fillna(0)
    df = df[df['MEASUREDLENGTH'] > min_length]
    df = df.sort_values('MEASUREDLENGTH', ascending=False)

# --- SWNO MULTIselect ---
unique_swnos = sorted(df['SWNO'].astype(str).unique().tolist())
swno_options = ["All"] + unique_swnos
selected_swnos = st.multiselect("Select SWNO(s) to visualize:", swno_options, default=["All"])
if "All" not in selected_swnos:
    df = df[df['SWNO'].astype(str).isin(selected_swnos)]
    if df.empty:
        st.warning("No cable with these SWNOs found after filtering.")
        st.stop()

# --- Remove unwanted REMARKS ---
df = df[~df['REMARKS'].str.upper().isin(['ABANDONED', 'DISCONNECTED'])]

# --- Standardize station names ---
station_standardization_map = {
    # ... (your map, unchanged)
    'MAROL': 'MAROL REC-STN',
    'SAHAR PLAZA': 'SAHAR PLAZA REC-STN',
    'MAHANANDA': 'MAHANANDA REC-STN',
    '220kV CHEMBUR REC-STN': 'CHEMBUR REC-STN',
    '220KV CHEMBUR REC-STN': 'CHEMBUR REC-STN',
    "TATA'S CHEMBUR REC-STN": 'CHEMBUR REC-STN',
    'CHEMBUR REC-STN': 'CHEMBUR REC-STN',
    "TATA'S BORIVLI REC-STN": 'BORIVALI REC-STN',
    '220kV BORIVALI REC-STN': 'BORIVALI REC-STN',
    'DAHISAR WEST REC-STN': 'DAHISAR REC-STN',
    'DAHISAR CHECKNAKA REC-STN': 'DAHISAR REC-STN',
    'NIRLON B10 RECEIVING STATION': 'NIRLON REC-STN',
    'KANAKIA CCI': 'KANAKIA CCI REC-STN',
    'SHANTI STAR MIRA REC-STN': 'BHAYANDAR REC-STN',
    'BHAYANDAR WEST REC-STN': 'BHAYANDAR REC-STN',
    'SAMBHAJI NAGAR': 'SAMBHAJI NAGAR REC-STN',
    'DINDOSHI VIA OMKAR REC STN': 'DINDOSHI REC-STN',
    'OMKAR RECEIVING STATION': 'DINDOSHI REC-STN',
    'NAHAR SHAKTI R/S': 'NAHAR SHAKTI REC-STN',
    'JUHU NORTH REC-STN': 'JUHU REC-STN',
    'ANDHERI': 'ANDHERI REC-STN',
    'AMBIVLI': 'AMBIVLI REC-STN',
    'KALPATARU LBS R/S': 'KALPATARU LBS REC-STN',
    'NETMAGIC DC-9 NO 1 BUS 1 DSS': 'NETMAGIC DC-9 NO 1',
    'NETMAGIC DC-9 NO 1 RECEIVING STATIO': 'NETMAGIC DC-9 NO 1',
    'PALM COURT': 'PALM COURT REC-STN',
    '220 kV AAREY EHV STATION': 'AAREY 220KV R/S',
    "TATA'S SAKI REC-STN": 'SAKI REC-STN',
    '220kV SAKI REC-STN': 'SAKI REC-STN',
    "TATA'S DHARAVI REC-STN": 'TATA DHARAVI REC-STN',
    "TATA's DHARAVI REC-STN": 'TATA DHARAVI REC-STN',
    '220kV VERSOVA REC-STN': 'VERSOVA REC-STN',
    "TATA'S VERSOVA REC-STN": 'VERSOVA REC-STN',
    '220kV GORAI REC-STN': 'GORAI REC-STN'
}
df['SOURCE_SS'] = df['SOURCE_SS'].map(station_standardization_map).fillna(df['SOURCE_SS'])
df['DESTINATION_SS'] = df['DESTINATION_SS'].map(station_standardization_map).fillna(df['DESTINATION_SS'])

st.markdown(f"**Unique SOURCE_SS count after standardization:** {df['SOURCE_SS'].nunique()}")
st.markdown(f"**Unique DESTINATION_SS count after standardization:** {df['DESTINATION_SS'].nunique()}")

# --- Advanced Health Score Function ---
def sigmoid(x, k=5):
    return 1 / (1 + np.exp(-k * (x - 0.5)))

def advanced_health_score(df):
    weights = {
        'Num_Faults': 0.0402,
        'CURRENT_MEAN': 0.1368,
        'CURRENT_STD': 0.0269,
        'CABLE_AGE': 0.2185,
        'PEAK_STD': 0.0284,
        'CYCLE_STD': 0.1075,
        'OVR_MEAN': 0.0274,
        'OVR_STD': 0.0329,
        'FUSE_COUNT': 0.024,
        'CABLETYPE_RISK': 0.0391,
        'MATERIAL_RISK': 0.3182
    }
    now_year = datetime.now().year
    if 'DATECREATED' in df.columns:
        years = pd.to_datetime(df['DATECREATED'], errors='coerce').dt.year
        df['CABLE_AGE'] = now_year - years.fillna(now_year)
    current_cols = [c for c in df.columns if c.startswith('I_Month_')]
    if current_cols:
        currents = df[current_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df['CURRENT_MEAN'] = currents.mean(axis=1)
        df['CURRENT_STD'] = currents.std(axis=1)
    for prefix, name in [('PEAK_Month_', 'PEAK'), ('CYCLE_Month_', 'CYCLE'), ('OVR_Month_', 'OVR')]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            arr = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            df[f'{name}_MEAN'] = arr.mean(axis=1)
            df[f'{name}_STD'] = arr.std(axis=1)
    if 'CABLETYPE' in df.columns:
        df['CABLETYPE_RISK'] = df['CABLETYPE'].apply(lambda t: 1 if 'PILC' in str(t).upper() else (0.2 if 'XLPE' in str(t).upper() else 0.5))
    if 'CABLECONDUCTORMATERIAL' in df.columns:
        df['MATERIAL_RISK'] = df['CABLECONDUCTORMATERIAL'].apply(lambda m: 0.7 if 'AL' in str(m).upper() else (0.2 if 'CU' in str(m).upper() or 'COPPER' in str(m).upper() else 0.5))
    fuse_cols = [col for col in df.columns if col.startswith('FAULT_RELAY_FUSE')]
    df['FUSE_COUNT'] = df[fuse_cols].apply(pd.to_numeric, errors='coerce').fillna(0).sum(axis=1) if fuse_cols else 0
    used = {k: v for k, v in weights.items() if k in df.columns}
    total_weight = sum(used.values())
    score = pd.Series(0.0, index=df.index)
    top3_features = []
    for i in df.index:
        contributions = {}
        for k, w in used.items():
            x = pd.to_numeric(df[k], errors='coerce').fillna(0)
            if (x.max() - x.min()) != 0:
                x_normed = (x - x.min()) / (x.max() - x.min())
                sig = sigmoid(x_normed[i])
            else:
                sig = sigmoid(0.5)
            contributions[k] = sig * (w / total_weight)
        sorted_feats = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        score[i] = sum(contributions.values())
        top3_features.append(", ".join([f for f, _ in sorted_feats[:3]]))
    df['Health_Score'] = (1 + 9 * (1 - score)).round().astype(int)
    df['Top3_Contributors'] = top3_features
    return df

df = advanced_health_score(df)

def health_color(score, remarks):
    cmap = matplotlib.cm.get_cmap('RdYlGn')
    rgb = cmap((score - 1) / 9)[:3]
    return '#{:02x}{:02x}{:02x}'.format(*(int(255 * x) for x in rgb))


# --- Network Visualization ---
G = nx.MultiDiGraph()
for idx, row in df.iterrows():
    src = row["SOURCE_SS"]
    dst = row["DESTINATION_SS"]
    swno = row['SWNO'] if 'SWNO' in row else f"edge_{idx}"
    health_score = int(row['Health_Score'])
    edge_color = health_color(health_score, row.get('REMARKS', ' '))
    # Show health score / remarks as the EDGE LABEL
    # Prepare REMARKS for label (first 2 uppercase letters or empty)
    remarks1 = str(row.get('REMARKS', '')).strip().upper()
    remarks = str(row.get('REMARKS', '')).strip().upper()
    remarks_label = remarks[:2] if remarks and remarks != '-' else ''

    if remarks_label:
        label = f"{health_score} / {remarks_label}"
    else:
        label = f"{health_score}"

    
    if viz_mode == "With Comments":
        main_col = 'COMMENTS'
        main_text = str(row.get(main_col, '-'))
        wrapped_text = "\n".join(textwrap.wrap(main_text, width=60))
        details = f"Comments: {wrapped_text}"
    else:
        main_col = 'PATH'
        main_text = str(row.get(main_col, '-'))
        wrapped_text = "\n".join(textwrap.wrap(main_text, width=60))
        details = f"Path: {wrapped_text}"
    
    tooltip = "\n".join([
        f"SWNO: {swno}",
        f"Top Factors: {row.get('Top3_Contributors', '-')}",
        f"Faults: {row.get('Num_Faults','-')}",
        f"Cable Type: {row.get('CABLETYPE','-')}",
        f"Measured Length: {row.get('MEASUREDLENGTH','-')}",
        f"remarks: {remarks1}",
        details
    ])

    for node in [src, dst]:
        if node not in G:
            G.add_node(node, title=f"Substation: {node}", color="#7FB3D5")

    G.add_edge(src, dst, color=edge_color, label=label, title=tooltip, width=3 + health_score / 2)


net = Network(notebook=False, directed=True, width="100%", height="1000px", bgcolor="#fff")
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
        "maxVelocity": 100,
        "solver": "forceAtlas2Based",
        "timestep": 0.60,
        "stabilization": {"enabled": true, "iterations": 80 }
    }
    }
    """)

with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
    net.save_graph(temp_file.name)
    temp_file.seek(0)
    html_content = temp_file.read().decode()
    html_content += """
    <script type="text/javascript">
    setTimeout(function() {
        if (window.network) {
        network.setOptions({physics: false});
        }
    }, 30000);
    </script>
    """
    st.components.v1.html(html_content, height=1000, scrolling=True)

# --- Summary ---
st.markdown("### Health Score Distribution")
st.bar_chart(df['Health_Score'].value_counts().sort_index())

if viz_mode == "With Comments":
    showcols = ['SWNO','SOURCE_SS','DESTINATION_SS','Health_Score','Top3_Contributors','Num_Faults','COMMENTS']
else:
    showcols = ['SWNO','SOURCE_SS','DESTINATION_SS','Health_Score','Top3_Contributors','Num_Faults','PATH']
showcols += [col for col in ['CURRENT_MEAN','CURRENT_STD','CABLE_AGE'] if col in df.columns]
showcols = [c for c in showcols if c in df.columns]
st.markdown("### Top 10 Unhealthy Cables")
st.dataframe(df.sort_values('Health_Score').head(10)[showcols])
st.markdown("### All Cable Health Table")
st.dataframe(df[showcols].sort_values('Health_Score'))

st.markdown("#### Health Score Color Legend (1=Red, 10=Green)")
fig, ax = plt.subplots(figsize=(5, 0.4))
cmap = matplotlib.cm.get_cmap('RdYlGn')
gradient = np.linspace(0, 1, 256).reshape(1, -1)
ax.imshow(gradient, aspect='auto', cmap=cmap)
ax.set_axis_off()
ax.set_title("1 (Worst)      →      10 (Best)", fontsize=12)
st.pyplot(fig, use_container_width=True)
