import streamlit as st
import sys
import os
import plotly.graph_objects as go
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.main import full_analysis

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="CodeSense AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state initialization ──────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CodeSense AI")
    st.divider()

    st.markdown("### 📊 Session Stats")
    st.metric("Analyses Run", st.session_state.analysis_count)
    st.metric("Saved in History", len(st.session_state.history))

    st.divider()
    st.markdown("### 🎯 Quick Examples")

    if st.button("✅ Load Good Code", use_container_width=True):
        st.session_state.example_code = """def calculate_total_price(items, tax_rate):
    # Calculate total price including tax
    if not items:
        raise ValueError("Items list cannot be empty")
    if tax_rate < 0:
        raise ValueError("Tax rate cannot be negative")
    subtotal = sum(item['price'] for item in items)
    tax_amount = subtotal * tax_rate
    total = subtotal + tax_amount
    return round(total, 2)"""

    if st.button("❌ Load Bad Code", use_container_width=True):
        st.session_state.example_code = """def f(x,y,z):
    a=x+y
    b=a*z
    return b"""

    if st.button("🟡 Load Medium Code", use_container_width=True):
        st.session_state.example_code = """def add_numbers(a, b):
    result = a + b
    return result"""

    st.divider()
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.analysis_count = 0
        st.success("History cleared!")

# ── Main header ───────────────────────────────────────────
st.markdown('<p class="main-header">🤖 CodeSense AI</p>', unsafe_allow_html=True)
st.markdown("AI-powered code review — get instant feedback on your code quality")
st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📜 History", "ℹ️ About"])

# ════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Your Code")

        default_code = ""
        if hasattr(st.session_state, 'example_code'):
            default_code = st.session_state.example_code

        code_input = st.text_area(
            label="paste code",
            value=default_code,
            placeholder="Paste your Python, JavaScript, or any code here...",
            height=320,
            label_visibility="collapsed"
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            analyze_btn = st.button(
                "🔍 Analyze My Code",
                type="primary",
                use_container_width=True
            )
        with col_btn2:
            clear_btn = st.button(
                "🗑️ Clear",
                use_container_width=True
            )

        if clear_btn:
            if hasattr(st.session_state, 'example_code'):
                del st.session_state.example_code
            st.rerun()

    with col2:
        st.subheader("📊 Results")

        if analyze_btn and code_input.strip():
            with st.spinner("🤖 Analyzing your code..."):
                result = full_analysis(code_input)

            # Save to history
            st.session_state.history.append({
                "code": code_input,
                "result": result,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "date": datetime.now().strftime("%Y-%m-%d")
            })
            st.session_state.analysis_count += 1

            # ── Score gauge chart ──────────────────────────
            score = result['quality_score']

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Quality Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ff4b4b"},
                        {'range': [40, 70], 'color': "#ffa500"},
                        {'range': [70, 100], 'color': "#00cc44"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            # ── Prediction badge ───────────────────────────
            pred = result['prediction']
            conf = result['confidence']

            if pred == "good":
                st.success(f"✅ **GOOD CODE** — {conf}% confident")
            else:
                st.error(f"❌ **NEEDS IMPROVEMENT** — {conf}% confident")

            # ── Stats row ──────────────────────────────────
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Lines", result['total_lines'])
            s2.metric("Comments", result['comment_lines'])
            s3.metric("Issues", result['issues_count'])
            s4.metric("Score", f"{score}/100")

            # ── Issues ─────────────────────────────────────
            # ── Syntax error — show prominently ───────────────────────
            if result.get('has_syntax_error'):
                 st.error(f"🚨 **SYNTAX ERROR DETECTED**")
                 st.code(result['syntax_error_detail'], language="text")
                 
            if result['issues_found']:
                st.markdown("**⚠️ Issues:**")
                for issue in result['issues_found']:
                    st.warning(f"⚠️ {issue}")

            # ── Suggestions ────────────────────────────────
            if result['suggestions']:
                st.markdown("**💡 Suggestions:**")
                for suggestion in result['suggestions']:
                    st.info(f"💡 {suggestion}")

            # ── Code preview with syntax highlighting ──────
            with st.expander("👁️ View Analyzed Code"):
                st.code(code_input, language="python")

        elif analyze_btn and not code_input.strip():
            st.warning("⚠️ Please paste some code first!")

        else:
            st.markdown("""
            ### 👈 How to use:
            1. Paste your code in the left panel
            2. Click **Analyze My Code**
            3. Get instant AI feedback!

            Or load an example from the **sidebar** →
            """)

# ════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("📜 Analysis History")

    if not st.session_state.history:
        st.info("No analyses yet. Go to the Analyze tab and review some code!")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            score = item['result']['quality_score']
            pred = item['result']['prediction']

            if score >= 70:
                icon = "🟢"
            elif score >= 40:
                icon = "🟡"
            else:
                icon = "🔴"

            with st.expander(
                f"{icon} Analysis #{len(st.session_state.history) - i} "
                f"— Score: {score}/100 — {item['timestamp']}"
            ):
                h_col1, h_col2 = st.columns(2)

                with h_col1:
                    st.markdown(f"**Prediction:** {pred.upper()}")
                    st.markdown(f"**Confidence:** {item['result']['confidence']}%")
                    st.markdown(f"**Issues found:** {item['result']['issues_count']}")

                    if item['result']['issues_found']:
                        for issue in item['result']['issues_found']:
                            st.warning(f"⚠️ {issue}")

                with h_col2:
                    st.markdown("**Code analyzed:**")
                    st.code(
                        item['code'][:300] + "..." if len(item['code']) > 300 else item['code'],
                        language="python"
                    )

# ════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ About CodeSense AI")

    st.markdown("""
    ## What is CodeSense AI?

    CodeSense AI is an intelligent code review tool that uses
    **Machine Learning** to analyze your code quality instantly.

    ## 🔧 How it works

    1. **TF-IDF Vectorizer** converts your code into numbers
    2. **Random Forest** (100 decision trees) predicts quality
    3. **Rule-based checks** detect specific issues
    4. **Suggestions engine** tells you exactly how to fix problems

    ## 🛠️ Tech Stack

    | Component | Technology |
    |---|---|
    | ML Model | Random Forest (sklearn) |
    | Vectorizer | TF-IDF |
    | UI | Streamlit |
    | Charts | Plotly |
    | Language | Python |

    ## 📈 Coming Soon
    - CodeBERT transformer model (Week 2)
    - RAG-powered detailed reviews (Week 3)
    - React frontend (Week 4)
    - Live deployment (Week 4)
    """)