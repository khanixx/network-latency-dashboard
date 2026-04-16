import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import google.generativeai as genai

st.set_page_config(page_title="RIPE Atlas Multi-Provider", page_icon="📊", layout="wide")

@st.cache_data(ttl=600)
def fetch_probe_data(measurement_id: int, probe_id: int, provider_name: str, days: int) -> pd.DataFrame:
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=days)).timestamp())
    
    url = f"https://atlas.ripe.net/api/v2/measurements/{measurement_id}/results/"
    params = {"probe_ids": probe_id, "start": start_time, "stop": end_time}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        records = [
            {
                "Time": datetime.fromtimestamp(entry['timestamp']), 
                "RTT (ms)": entry['min'],
                "Provider": provider_name
            }
            for entry in data if 'min' in entry
        ]
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Error fetching Probe {probe_id}: {e}")
        return pd.DataFrame()

def generate_ai_report(summary_df: pd.DataFrame, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert senior network engineer. I have collected latency data (RTT in milliseconds) from different ISPs to the same target over several days.
    
    Here are the summary statistics:
    {summary_df.to_markdown(index=False)}
    
    Your task:
    1. Analyze this data.
    2. Compare the providers against each other.
    3. Evaluate network stability (jitter) based on Min, Avg, and Max RTT differences.
    4. Provide a concise, professional conclusion and recommendation in a report format.
    """
    
    response = model.generate_content(prompt)
    return response.text

def main():
    st.title("📊 Multi-Provider Latency Monitor")
    st.markdown("Enter the required IDs to fetch and visualize RTT data.")
    
    with st.sidebar:
        st.header("⚙️ Query Settings")
        measurement_id = st.number_input("Measurement ID", value=None, placeholder="e.g., 86227489", step=1)
        days = st.slider("History Period (days)", min_value=1, max_value=14, value=3)
        
        st.subheader("📡 Probes Configuration")
        probes_input = st.text_area(
            "Format -> ProbeID: Provider Name", 
            value="", 
            placeholder="7734: Kazakhtelecom (AS9198)\n53961: Beeline KZ (AS21299)\n6746: Tele2 / Altel (AS48503)",
            height=120
        )
        
        st.divider()
        st.subheader("🤖 AI Analytics")
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your key here...")
        
        submit_button = st.button("Load / Update Data", type="primary", width="stretch")

    if submit_button:
        if measurement_id is None:
            st.warning("Please enter a valid Measurement ID.")
            return
            
        if not probes_input.strip():
            st.warning("Please provide at least one probe in the text area.")
            return

        probes_dict = {}
        for line in probes_input.strip().split('\n'):
            if ':' in line:
                p_id, p_name = line.split(':', 1)
                try:
                    probes_dict[int(p_id.strip())] = p_name.strip()
                except ValueError:
                    continue
                
        if not probes_dict:
            st.warning("Invalid probe format. Please use 'ID: Name'.")
            return

        all_dfs = []
        with st.spinner('Fetching data from RIPE Atlas...'):
            for p_id, p_name in probes_dict.items():
                df = fetch_probe_data(measurement_id, p_id, p_name, days)
                if not df.empty:
                    all_dfs.append(df)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            
            st.subheader("📈 Summary Statistics")
            summary = final_df.groupby("Provider")["RTT (ms)"].agg(['count', 'min', 'mean', 'max']).reset_index()
            summary.columns = ["Provider", "Total Points", "Min (ms)", "Avg (ms)", "Max (ms)"]
            st.dataframe(
                summary.style.format({"Min (ms)": "{:.1f}", "Avg (ms)": "{:.1f}", "Max (ms)": "{:.1f}"}), 
                width="stretch"
            )
            
            st.subheader("🤖 AI Network Analysis")
            if not api_key:
                st.info("💡 Enter your Gemini API Key in the sidebar to generate an automated expert report.")
            else:
                with st.spinner("AI is analyzing the network data..."):
                    try:
                        report = generate_ai_report(summary, api_key)
                        st.success(report) 
                    except Exception as e:
                        st.error(f"Failed to generate report. Check your API key. Error: {e}")
            
            st.divider()
            
            st.subheader("Figure 1 — RTT Overlay (Cross-provider comparison)")
            fig_line = px.line(
                final_df, 
                x="Time", 
                y="RTT (ms)", 
                color="Provider",
                template="plotly_white",
                color_discrete_sequence=['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']
            )
            fig_line.update_layout(
                xaxis_title="Local Time",
                yaxis_title="RTT (ms)",
                hovermode="x unified", 
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_line, width="stretch")
            
            st.divider()
            
            st.subheader("Figure 2 — RTT Distribution (Boxplot)")
            fig_box = px.box(
                final_df, 
                x="Provider", 
                y="RTT (ms)", 
                color="Provider",
                template="plotly_white",
                color_discrete_sequence=['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']
            )
            fig_box.update_layout(
                xaxis_title="",
                yaxis_title="RTT (ms)",
                margin=dict(l=0, r=0, t=20, b=0), 
                showlegend=False
            )
            st.plotly_chart(fig_box, width="stretch")
            
            with st.expander("📊 View Raw Data"):
                st.dataframe(final_df, width="stretch")
        else:
            st.warning("No data retrieved for the specified probes.")

if __name__ == "__main__":
    main()