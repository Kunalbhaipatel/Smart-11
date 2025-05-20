import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Shaker Health Dashboard", layout="wide")

# Google-style CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-color: #f5f5f5;
    }
    .main {
        background-color: #ffffff;
        padding: 1rem 2rem;
        border-radius: 8px;
        margin: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #202124;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #4285F4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stSlider>div>div {
        background-color: #e8f0fe !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f1f3f4;
        border-radius: 8px;
        padding: 0.5rem;
    }
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar Branding
try:
    st.sidebar.image("assets/Prodigy_IQ_logo.png", width=200)
except:
    st.sidebar.warning("⚠️ Logo failed to load.")

st.title("🛠️ Real-Time Shaker Monitoring Dashboard")

# Inputs
SCREEN_MESH_CAPACITY = {"API 100": 250, "API 140": 200, "API 170": 160, "API 200": 120}
df_mesh_type = st.sidebar.selectbox("Select Screen Mesh Type", list(SCREEN_MESH_CAPACITY.keys()))
mesh_capacity = SCREEN_MESH_CAPACITY[df_mesh_type]
util_threshold = st.sidebar.slider("Utilization Threshold (%)", 50, 100, 80)

uploaded_file = st.file_uploader("📤 Upload Shaker CSV Data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    try:
        df['Timestamp'] = pd.to_datetime(df['YYYY/MM/DD'] + ' ' + df['HH:MM:SS'])
        df = df.sort_values('Timestamp')
        df['Date'] = df['Timestamp'].dt.date
    except:
        st.warning("Could not parse timestamp.")

    if 'Screen Utilization (%)' not in df.columns and 'Weight on Bit (klbs)' in df.columns:
        df['Solids Volume Rate (gpm)'] = df['Weight on Bit (klbs)'] * df['MA_Flow_Rate (gal/min)'] / 100
        df['Screen Utilization (%)'] = (df['Solids Volume Rate (gpm)'] / mesh_capacity) * 100

    st.subheader("📊 Key Metrics")
    avg_util = df['Screen Utilization (%)'].mean()
    avg_flow = df['MA_Flow_Rate (gal/min)'].mean()
    shaker_max = df['SHAKER #3 (PERCENT)'].max()
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Utilization", f"{avg_util:.1f}%")
    col2.metric("Avg Flow Rate", f"{avg_flow:.1f} gpm")
    col3.metric("Max SHKR3", f"{shaker_max:.1f}%")

    tabs = st.tabs(["📈 Charts", "📋 Data"])
    with tabs[0]:
        st.markdown("#### Shaker Output Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SHAKER #1 (Units)'], name="SHAKER #1"))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SHAKER #2 (Units)'], name="SHAKER #2"))
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['SHAKER #3 (PERCENT)'], name="SHAKER #3"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Daily Utilization Summary")
        daily_avg = df.groupby('Date').agg({
            'Screen Utilization (%)': 'mean',
            'MA_Flow_Rate (gal/min)': 'mean',
            'SHAKER #3 (PERCENT)': ['mean', 'max']
        }).reset_index()
        daily_avg.columns = ['Date', 'Avg Utilization', 'Avg Flow Rate', 'Avg SHKR3', 'Max SHKR3']
        daily_avg['Exceeds Threshold'] = daily_avg['Avg Utilization'] > util_threshold

        fig2 = px.bar(daily_avg, x='Date', y='Avg Utilization', color='Exceeds Threshold',
                      color_discrete_map={True: '#ea4335', False: '#34a853'})
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[1]:
        st.dataframe(df)

else:
    st.info("📁 Upload a valid shaker CSV file to get started.")

# Footer
st.markdown("""
    <div style='text-align: center; padding: 20px; font-size: 13px; color: #888'>
        © 2024 Prodigy IQ • Clean UI Inspired by Google
    </div>
""", unsafe_allow_html=True)
