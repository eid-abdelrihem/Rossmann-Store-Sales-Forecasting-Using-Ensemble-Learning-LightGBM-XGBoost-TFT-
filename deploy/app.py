"""
Rossmann Sales Intelligence System
===================================
Light & modern UI · Groq LLM · TFT + XGBoost Ensemble
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, base64
from groq import Groq

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Rossmann Sales Intelligence",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Light Modern CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg: #f8fafc;
    --card: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text2: #64748b;
    --accent: #6366f1;
    --accent-light: #eef2ff;
    --rose: #e11d48;
    --rose-light: #fff1f2;
    --emerald: #059669;
    --emerald-light: #ecfdf5;
    --amber: #d97706;
    --amber-light: #fffbeb;
    --blue: #2563eb;
    --blue-light: #eff6ff;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.05);
    --radius: 16px;
}

html, body, [data-testid="stApp"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }

/* Sidebar */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #fefefe 0%, #f1f5f9 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] h2 {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
    font-weight: 700;
}

/* Banner */
.banner-container {
    position: relative;
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-lg);
}
.banner-img {
    width: 100%;
    height: 200px;
    object-fit: cover;
    display: block;
}
.banner-overlay {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 1.5rem 2rem;
    background: linear-gradient(transparent, rgba(0,0,0,0.7));
}
.banner-overlay h1 {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.banner-overlay p {
    color: rgba(255,255,255,0.85);
    font-size: 0.9rem;
    margin: 4px 0 0;
}

/* KPI Cards */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-bottom: 1.5rem;
}
.kpi {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    box-shadow: var(--shadow);
    transition: all 0.25s ease;
}
.kpi:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: var(--accent);
}
.kpi .icon { font-size: 1.5rem; margin-bottom: 4px; }
.kpi .val {
    font-size: 1.5rem;
    font-weight: 800;
    line-height: 1.2;
}
.kpi .lbl {
    font-size: 0.72rem;
    color: var(--text2);
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}
.kpi.indigo .val { color: var(--accent); }
.kpi.rose .val { color: var(--rose); }
.kpi.emerald .val { color: var(--emerald); }
.kpi.amber .val { color: var(--amber); }
.kpi.blue .val { color: var(--blue); }

/* Sidebar profile card */
.profile-card {
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3);
    margin: 0.8rem 0;
}
.profile-card .big { font-size: 2rem; font-weight: 800; }
.profile-card .sub { font-size: 0.8rem; opacity: 0.85; margin-top: 2px; }

/* Sidebar tips */
.tips-card {
    background: var(--accent-light);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 0.9rem;
    margin-top: 0.5rem;
}
.tips-card .tip {
    font-size: 0.8rem;
    color: var(--accent);
    padding: 3px 0;
    cursor: pointer;
}
.tips-card .tip:hover { text-decoration: underline; }

/* Chat section */
.chat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.5rem;
}
.chat-header h3 {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 700;
}
.chat-badge {
    background: var(--emerald-light);
    color: var(--emerald);
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

div[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    border: 1px solid var(--border);
}

/* Tabs */
div[data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_daily():
    df = pd.read_csv(os.path.join(BASE_DIR, "models", "llm_daily_context.csv"))
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_profiles():
    with open(os.path.join(BASE_DIR, "models", "store_profiles.json"), "r") as f:
        return json.load(f)

def get_banner_base64():
    path = os.path.join(BASE_DIR, "assets", "banner.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Senior Retail Sales Analyst for Rossmann drugstores.
You analyze store performance using an ML ensemble (TFT deep learning + XGBoost + Ridge meta-learner).

Your modes:
1. **Explain** — Why sales changed. Cite exact €amounts, dates, promo/holiday flags.
2. **Recommend** — Promo timing, staffing, inventory advice based on patterns.
3. **Compare** — Promo vs non-promo, weekday comparisons, model accuracy.

Rules:
- ALWAYS cite numbers: €amounts, dates, percentages from the data.
- Use bullet points, max 5 key insights.
- Use € currency. Mention Error% and confidence when relevant.
- End with a **bold actionable takeaway**.
- Be friendly and professional.
"""

def build_ctx(sid, profiles, df):
    p = profiles.get(str(sid), {})
    sd = df[df["Store"] == sid].sort_values("Date")
    day_map = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
    
    c = f"""# Store {sid}
Type: {p.get('store_type','?').upper()} | Assortment: {p.get('assortment','?').upper()}
Avg: €{p.get('avg_daily_sales',0):,.0f} | Median: €{p.get('median_sales',0):,.0f} | Max: €{p.get('max_sales',0):,.0f}
Customers: {p.get('avg_customers',0):,.0f}/day | Open days: {p.get('total_open_days',0)}
Best: {p.get('best_day','?')} | Worst: {p.get('worst_day','?')}
Days: {json.dumps(p.get('sales_by_day',{}))}
Competition: {p.get('competition_dist','?')}m (since {p.get('competition_open_since','?')})
Promo lift: +{p.get('promo_lift_pct',0)}% (€{p.get('avg_sales_with_promo',0):,.0f} vs €{p.get('avg_sales_no_promo',0):,.0f})
Promo2: {p.get('has_promo2',False)} | Interval: {p.get('promo_interval','—')}
Holiday avg: €{p.get('avg_sales_holiday','N/A')} | School: €{p.get('avg_sales_school_holiday','N/A')}
RMSPE: TFT={p.get('tft_rmspe','?')} | XGB={p.get('xgb_rmspe','?')} | Ensemble={p.get('ensemble_rmspe','?')}

# Daily Data
Date|Day|Actual|Ens|Err%|Promo|Hol|SchHol|Lag7|Mean7|DowMean
"""
    for _, r in sd.iterrows():
        c += (f"{r['Date'].strftime('%m-%d')}|{day_map.get(int(r.get('DayOfWeek',0)),'?')}|"
              f"{r['Actual']:.0f}|{r['Ensemble_Pred']:.0f}|{r.get('Error_Pct','?')}|"
              f"{'Y' if r.get('Promo',0)==1 else 'N'}|"
              f"{'Y' if r.get('StateHoliday',0) not in [0,'0'] else 'N'}|"
              f"{'Y' if r.get('SchoolHoliday',0)==1 else 'N'}|"
              f"{r.get('Sales_Lag_7','?')}|{r.get('Sales_Mean_7','?')}|"
              f"{r.get('store_dow_mean_sales','?')}\n")
    return c

def ask_llm(key, msgs, ctx):
    client = Groq(api_key=key)
    full = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + ctx}]
    full.extend(msgs)
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile", messages=full,
        temperature=0.3, max_tokens=1500)
    return r.choices[0].message.content

# ──────────────────────────────────────────────
# Charts (Light Theme)
# ──────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="white",
    plot_bgcolor="#fafbfc",
    font=dict(family="Plus Jakarta Sans", color="#334155"),
    margin=dict(l=50, r=20, t=60, b=30),
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center",
                bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
)

def chart_forecast(sd, sid):
    fig = go.Figure()
    # Confidence band
    fig.add_trace(go.Scatter(x=sd["Date"], y=sd["TFT_Upper"], mode="lines",
        line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=sd["Date"], y=sd["TFT_Lower"], mode="lines",
        line=dict(width=0), fill="tonexty", fillcolor="rgba(99,102,241,0.1)",
        name="95% Confidence"))
    # Actual Sales
    fig.add_trace(go.Scatter(x=sd["Date"], y=sd["Actual"], mode="lines+markers",
        name="Actual Sales", line=dict(color="#1e293b", width=3),
        marker=dict(size=6, color="#1e293b")))
    # Ensemble Only
    fig.add_trace(go.Scatter(x=sd["Date"], y=sd["Ensemble_Pred"], mode="lines+markers",
        name="Ensemble Prediction", line=dict(color="#6366f1", width=3),
        marker=dict(size=6, color="#6366f1")))
    # Promo markers
    promo = sd[sd.get("Promo", pd.Series(dtype=float)) == 1]
    if len(promo) > 0:
        fig.add_trace(go.Scatter(x=promo["Date"], y=promo["Actual"], mode="markers",
            name="🏷️ Promo Day", marker=dict(color="#e11d48", size=12, symbol="diamond",
            line=dict(width=1.5, color="white"))))
    fig.update_layout(**CHART_LAYOUT, height=430,
        title=dict(text=f"📈 Store {sid} — Ensemble Forecast vs Actual", font=dict(size=16, weight=700)))
    return fig

def chart_error(sd, sid):
    colors = ['#ef4444' if abs(e) > 10 else ('#f59e0b' if abs(e) > 5 else '#10b981')
              for e in sd['Error_Pct'].fillna(0)]
    fig = go.Figure(go.Bar(x=sd["Date"], y=sd["Error_Pct"], marker_color=colors))
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1.5)
    fig.add_hline(y=10, line_dash="dot", line_color="#fca5a5", annotation_text="+10%")
    fig.add_hline(y=-10, line_dash="dot", line_color="#fca5a5", annotation_text="-10%")
    fig.update_layout(**CHART_LAYOUT, height=320,
        title=dict(text=f"📊 Store {sid} — Prediction Error %", font=dict(size=16, weight=700)))
    return fig

def chart_dow(p):
    sbd = p.get("sales_by_day", {})
    if not sbd: return None
    days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    vals = [sbd.get(d, 0) for d in days]
    colors = []
    for v in vals:
        if v == max(vals): colors.append("#059669")
        elif v == min(vals): colors.append("#e11d48")
        else: colors.append("#6366f1")
    fig = go.Figure(go.Bar(x=days, y=vals, marker_color=colors,
        marker_line=dict(width=0),
        text=[f"€{v:,.0f}" for v in vals], textposition="outside",
        textfont=dict(size=12, color="#475569", weight=600)))
    fig.update_layout(**CHART_LAYOUT, height=320,
        title=dict(text="📅 Average Sales by Day", font=dict(size=16, weight=700)))
    return fig

# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────
def main():
    df = load_daily()
    profiles = load_profiles()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🔐 API Key")
        key = st.text_input("Groq API Key:", type="password", key="gk",
            help="Free at console.groq.com/keys")
        if not key:
            st.info("🔗 [Get free key](https://console.groq.com/keys)")

        st.markdown("---")
        st.markdown("## 🏪 Select Store")
        stores = sorted(df["Store"].unique())
        sid = st.selectbox("Store:", stores, index=0, label_visibility="collapsed")

        p = profiles.get(str(sid), {})
        if p:
            st.markdown(f"""
            <div class="profile-card">
                <div class="big">€{p.get('avg_daily_sales',0):,.0f}</div>
                <div class="sub">Average Daily Sales</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("📅 Best", p.get("best_day", "?"))
            c2.metric("🏷️ Type", p.get("store_type", "?").upper())
            c1.metric("📈 Lift", f"+{p.get('promo_lift_pct',0)}%")
            c2.metric("🎯 Acc.", f"{(1-(p.get('ensemble_rmspe',0) or 0))*100:.1f}%")

        st.markdown("---")
        st.markdown("## 💬 Try Asking")
        st.markdown("""
        <div class="tips-card">
            <div class="tip">💬 Why did sales drop on July 6?</div>
            <div class="tip">💬 When should I run promotions?</div>
            <div class="tip">💬 Compare promo vs non-promo days</div>
            <div class="tip">💬 Best day to restock inventory?</div>
            <div class="tip">💬 Why was the model wrong on July 25?</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Banner ──
    b64 = get_banner_base64()
    if b64:
        st.markdown(f"""
        <div class="banner-container">
            <img src="data:image/png;base64,{b64}" class="banner-img"/>
            <div class="banner-overlay">
                <h1>🏪 Rossmann Sales Intelligence</h1>
                <p>AI-Powered Analysis · TFT + XGBoost Ensemble · {len(stores):,} Stores</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#6366f1,#8b5cf6,#a855f7);
            border-radius:16px;padding:2rem 2.5rem;margin-bottom:1.5rem;color:white;">
            <h1 style="margin:0;font-size:1.8rem;font-weight:800;">🏪 Rossmann Sales Intelligence</h1>
            <p style="margin:4px 0 0;opacity:0.9;">AI-Powered · TFT + XGBoost Ensemble · {len(stores):,} Stores</p>
        </div>
        """, unsafe_allow_html=True)

    sd = df[df["Store"] == sid].sort_values("Date")
    if len(sd) == 0:
        st.error(f"No data for Store {sid}")
        return

    # ── KPIs ──
    avg_err = sd["Error_Pct"].abs().mean()
    conf = sd["TFT_Confidence_Width"].mean()
    err_cls = "emerald" if avg_err < 5 else ("amber" if avg_err < 10 else "rose")
    
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi indigo">
            <div class="icon">🏪</div>
            <div class="val">Store {sid}</div>
            <div class="lbl">Type {p.get('store_type','?').upper()} · Assort. {p.get('assortment','?').upper()}</div>
        </div>
        <div class="kpi blue">
            <div class="icon">💰</div>
            <div class="val">€{sd['Actual'].mean():,.0f}</div>
            <div class="lbl">Avg Actual Sales</div>
        </div>
        <div class="kpi indigo">
            <div class="icon">🤖</div>
            <div class="val">€{sd['Ensemble_Pred'].mean():,.0f}</div>
            <div class="lbl">Avg Prediction</div>
        </div>
        <div class="kpi {err_cls}">
            <div class="icon">{'✅' if avg_err < 5 else ('⚠️' if avg_err < 10 else '❌')}</div>
            <div class="val">{avg_err:.1f}%</div>
            <div class="lbl">Mean Abs Error</div>
        </div>
        <div class="kpi amber">
            <div class="icon">📏</div>
            <div class="val">±€{conf:,.0f}</div>
            <div class="lbl">95% Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts ──
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Error Analysis", "📅 Day-of-Week"])
    with tab1:
        st.plotly_chart(chart_forecast(sd, sid), use_container_width=True)
    with tab2:
        st.plotly_chart(chart_error(sd, sid), use_container_width=True)
    with tab3:
        fig = chart_dow(p)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # ── Chat ──
    st.markdown("---")
    st.markdown("""
    <div class="chat-header">
        <h3>💬 Ask the AI Analyst</h3>
        <span class="chat-badge">🟢 Powered by Llama 3.3</span>
    </div>
    """, unsafe_allow_html=True)

    if "msgs" not in st.session_state:
        st.session_state.msgs = []
    if "cs" not in st.session_state:
        st.session_state.cs = sid
    if st.session_state.cs != sid:
        st.session_state.msgs = []
        st.session_state.cs = sid

    for m in st.session_state.msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input(f"Ask about Store {sid}..."):
        st.session_state.msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not key:
            with st.chat_message("assistant"):
                msg = "⚠️ Please enter your Groq API key in the sidebar to start analyzing."
                st.warning(msg)
                st.session_state.msgs.append({"role": "assistant", "content": msg})
        else:
            with st.chat_message("assistant"):
                with st.spinner("🔍 Analyzing store data..."):
                    ctx = build_ctx(sid, profiles, df)
                    try:
                        ans = ask_llm(key, st.session_state.msgs, ctx)
                        st.markdown(ans)
                        st.session_state.msgs.append({"role": "assistant", "content": ans})
                    except Exception as e:
                        err_msg = f"❌ Analysis failed: {str(e)}"
                        st.error(err_msg)

if __name__ == "__main__":
    main()
