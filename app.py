import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

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

def generate_ai_report(summary_df: pd.DataFrame, api_key: str, provider: str) -> str:
    api_key = api_key.strip()
    
    prompt = f"""
    You are an expert senior network engineer. I have collected latency data (RTT in milliseconds) from different ISPs to the same target over several days.
    
    Here are the summary statistics:
    {summary_df.to_string(index=False)}
    
    Your task:
    1. Analyze this data.
    2. Compare the providers against each other.
    3. Evaluate network stability (jitter) based on Min, Avg, and Max RTT differences.
    4. Provide a concise, professional conclusion and recommendation in a report format.
    """
    
    if provider == "Google Gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        
        response = requests.post(url, headers=headers, json=data)
        
        if not response.ok:
            raise Exception(f"Google API Error {response.status_code}: {response.text}")
            
        return response.json()['candidates'][0]['content']['parts'][0]['text']
        
    elif provider == "OpenAI (ChatGPT)":
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if not response.ok:
            raise Exception(f"OpenAI API Error {response.status_code}: {response.text}")
            
        return response.json()['choices'][0]['message']['content']

    elif provider == "xAI (Grok)":
        url = "https://api.x.ai/v1/responses"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "grok-4.20-reasoning",
            "input": prompt
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if not response.ok:
            raise Exception(f"xAI API Error {response.status_code}: {response.text}")
            
        resp_json = response.json()
        try:
            for item in resp_json.get('output', []):
                content = item.get('content')
                if isinstance(content, list):
                    for c in content:
                        if c.get('type') == 'output_text':
                            return c.get('text')
                elif isinstance(content, str):
                    return content
            return str(resp_json)
        except Exception:
            return str(resp_json)

def main():
    st.title("📊 Multi-Provider Latency Monitor")
    st.markdown("Enter the required IDs to fetch and visualize RTT data. Leave blank to use default examples.")
    
    default_meas_id = 86227489
    default_probes_str = "7734: Kazakhtelecom (AS9198)\n53961: Beeline KZ (AS21299)\n6746: Tele2 / Altel (AS48503)"
    
    with st.sidebar:
        st.header("⚙️ Query Settings")
        measurement_id = st.number_input("Measurement ID", value=None, placeholder=f"e.g., {default_meas_id}", step=1)
        days = st.slider("History Period (days)", min_value=1, max_value=14, value=3)
        
        st.subheader("📡 Probes Configuration")
        probes_input = st.text_area(
            "Format -> ProbeID: Provider Name", 
            value="", 
            placeholder=default_probes_str,
            height=120
        )
        
        st.divider()
        st.subheader("🤖 AI Analytics")
        ai_provider = st.selectbox("Select AI Provider", ["Google Gemini", "OpenAI (ChatGPT)", "xAI (Grok)"])
        api_key = st.text_input("API Key", type="password", placeholder=f"Enter your {ai_provider.split()[0]} key...")
        
        submit_button = st.button("Load / Update Data", type="primary", width="stretch")
        
        st.divider()
        st.markdown("<div style='text-align: center; color: gray; font-size: 0.9em;'>👨‍💻 Made by Kamaliyev Abylaikhan (khanixx)</div>", unsafe_allow_html=True)

    if submit_button:
        active_meas_id = measurement_id if measurement_id is not None else default_meas_id
        active_probes_str = probes_input.strip() if probes_input.strip() else default_probes_str

        probes_dict = {}
        for line in active_probes_str.split('\n'):
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
                df = fetch_probe_data(active_meas_id, p_id, p_name, days)
                if not df.empty:
                    all_dfs.append(df)
        
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            colors = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']
            
            st.subheader("📈 Summary Statistics")
            summary = final_df.groupby("Provider")["RTT (ms)"].agg(['count', 'min', 'mean', 'max']).reset_index()
            summary.columns = ["Provider", "Total Points", "Min (ms)", "Avg (ms)", "Max (ms)"]
            st.dataframe(
                summary.style.format({"Min (ms)": "{:.1f}", "Avg (ms)": "{:.1f}", "Max (ms)": "{:.1f}"}), 
                width="stretch"
            )
            
            st.subheader("🤖 AI Network Analysis")
            if not api_key:
                st.info(f"💡 Enter your {ai_provider.split()[0]} API Key in the sidebar to generate an automated expert report.")
            else:
                with st.spinner(f"{ai_provider} is analyzing the network data..."):
                    try:
                        report = generate_ai_report(summary, api_key, ai_provider)
                        st.success(report) 
                    except Exception as e:
                        st.error(f"Analysis failed. Detail: {e}")
            
            st.divider()
            
            st.subheader("Figure 1 — RTT Overlay (Cross-provider comparison)")
            fig_line = px.line(
                final_df, 
                x="Time", 
                y="RTT (ms)", 
                color="Provider",
                template="plotly_white",
                color_discrete_sequence=colors
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
                color_discrete_sequence=colors
            )
            fig_box.update_layout(
                xaxis_title="",
                yaxis_title="RTT (ms)",
                margin=dict(l=0, r=0, t=20, b=0), 
                showlegend=False
            )
            st.plotly_chart(fig_box, width="stretch")

            st.divider()

            st.subheader("Figure 3 — RTT Frequency Histogram")
            fig_hist = px.histogram(
                final_df, 
                x="RTT (ms)", 
                color="Provider",
                barmode="overlay",
                template="plotly_white",
                opacity=0.6,
                color_discrete_sequence=colors
            )
            fig_hist.update_layout(
                xaxis_title="Latency (ms)",
                yaxis_title="Frequency",
                margin=dict(l=0, r=0, t=20, b=0)
            )
            st.plotly_chart(fig_hist, width="stretch")

            st.divider()

            st.subheader("Figure 4 — Individual Provider Timeseries")
            unique_providers = final_df['Provider'].unique()
            tabs = st.tabs(list(unique_providers))
            
            for i, provider in enumerate(unique_providers):
                with tabs[i]:
                    provider_df = final_df[final_df['Provider'] == provider]
                    fig_ind = px.line(
                        provider_df, 
                        x="Time", 
                        y="RTT (ms)",
                        template="plotly_white",
                        color_discrete_sequence=[colors[i % len(colors)]]
                    )
                    fig_ind.update_layout(
                        xaxis_title="Local Time",
                        yaxis_title="RTT (ms)",
                        hovermode="x unified",
                        margin=dict(l=0, r=0, t=20, b=0)
                    )
                    st.plotly_chart(fig_ind, width="stretch")
            
            st.divider()
            
            with st.expander("📊 View Raw Data"):
                st.dataframe(final_df, width="stretch")
        else:
            st.warning("No data retrieved for the specified probes.")

if __name__ == "__main__":
    main()