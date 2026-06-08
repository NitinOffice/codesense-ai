# frontend/app.py
import streamlit as st
import sys
import os

# Tell Python where to find backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.main import full_analysis

# ── Page configuration ────────────────────────────────────
st.set_page_config(
    page_title="CodeSense AI",
    page_icon="🤖",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────
st.title("🤖 CodeSense AI")
st.markdown("**AI-powered code review tool** — paste your code and get instant feedback")
st.divider()

# ── Layout: two columns ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Paste Your Code")
    
    code_input = st.text_area(
        label="Code input",
        placeholder="Paste your Python code here...",
        height=300,
        label_visibility="collapsed"
    )

analyze_button = st.button(
        "🔍 Analyze Code",
        type="primary",
        use_container_width=True
    )

# Example buttons
st.markdown("**Try an example:**")
ex_col1, ex_col2 = st.columns(2)
    
with ex_col1:
        good_example = st.button("✅ Good Code Example", use_container_width=True)
with ex_col2:
        bad_example = st.button("❌ Bad Code Example", use_container_width=True)

# Load examples when buttons clicked
if good_example:
        st.session_state.example_code = """
def calculate_total_price(items, tax_rate):
    # Calculate total price including tax
    if not items:
        raise ValueError("Items list cannot be empty")
    if tax_rate < 0:
        raise ValueError("Tax rate cannot be negative")
    subtotal = sum(item['price'] for item in items)
    tax_amount = subtotal * tax_rate
    total = subtotal + tax_amount
    return round(total, 2)
"""
if bad_example:
        st.session_state.example_code = """
def f(x,y,z):
    a=x+y
    b=a*z
    return b
"""

with col2:
    st.subheader("📊 Analysis Results")
    
    # Get code to analyze
    code_to_analyze = ""
    if hasattr(st.session_state, 'example_code'):
        code_to_analyze = st.session_state.example_code
    if code_input:
        code_to_analyze = code_input

if analyze_button and code_to_analyze:
        with st.spinner("🔍 Analyzing your code..."):
            result = full_analysis(code_to_analyze)

# Quality Score — big and prominent
        score = result['quality_score']
        if score >= 70:
            score_color = "🟢"
            score_label = "Good"
        elif score >= 40:
            score_color = "🟡"
            score_label = "Needs Work"
        else:
            score_color = "🔴"
            score_label = "Poor"

        st.metric(
            label="Quality Score",
            value=f"{score}/100",
            delta=score_label
        )

# Prediction confidence
        pred = result['prediction']
        conf = result['confidence']
        if pred == "good":
            st.success(f"✅ Prediction: **GOOD CODE** ({conf}% confident)")
        else:
            st.error(f"❌ Prediction: **BAD CODE** ({conf}% confident)")

# Basic stats
        st.markdown("**📈 Code Statistics:**")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Total Lines", result['total_lines'])
        with stat_col2:
            st.metric("Comment Lines", result['comment_lines'])
        with stat_col3:
            st.metric("Issues Found", result['issues_count'])

# Issues found
        if result['issues_found']:
            st.markdown("**⚠️ Issues Found:**")
            for issue in result['issues_found']:
                st.warning(f"⚠️ {issue}")
        else:
            st.success("✅ No major issues found!")

# Suggestions
        if result['suggestions']:
            st.markdown("**💡 How to Fix:**")
            for suggestion in result['suggestions']:
                st.info(f"💡 {suggestion}")


# Probability breakdown
        st.markdown("**🎯 Model Confidence:**")
        probs = result['all_probabilities']
        st.progress(probs.get('good', 0) / 100)
        st.caption(f"Good: {probs.get('good', 0)}%  |  Bad: {probs.get('bad', 0)}%")

elif analyze_button and not code_to_analyze:
        st.warning("⚠️ Please paste some code first!")
    
else:
        st.info("👈 Paste your code on the left and click **Analyze Code**")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("CodeSense AI — Built with sklearn, Random Forest, and Streamlit")