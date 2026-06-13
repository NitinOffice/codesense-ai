# 🤖 CodeSense AI

An AI-powered code review tool that analyzes your code quality 
instantly using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![sklearn](https://img.shields.io/badge/ML-scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Active-green)

---

## 🎯 What it does

Paste any code → get instant AI feedback:

- ✅ **Quality Score** (0-100) with visual gauge
- 🤖 **ML Prediction** — Good or Bad code with confidence %
- ⚠️ **Issue Detection** — missing comments, no error handling, bad variable names
- 💡 **Suggestions** — exactly how to fix each issue
- 📜 **History** — all your past analyses saved in session

---

## 🖥️ Screenshots

> Add screenshots here after running the app

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| ML Model | Random Forest (100 trees) | Predict code quality |
| Vectorizer | TF-IDF | Convert code to numbers |
| Rule Engine | Python regex | Detect specific issues |
| UI | Streamlit | Web interface |
| Charts | Plotly | Gauge visualization |
| Model Saving | joblib | Persist trained models |

---

## 📁 Project Structure

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/NitinOffice/codesense-ai.git
cd codesense-ai
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python backend/train_model.py
```

### 5. Run the app
```bash
streamlit run frontend/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 How the ML works

1. **Dataset** — 40 hand-labeled code examples (20 good, 20 bad)
2. **TF-IDF** — converts code text into numerical feature vectors
3. **Random Forest** — 100 decision trees vote on quality
4. **Rule checks** — regex-based detection of specific bad practices
5. **Suggestions** — mapped from detected issues to actionable fixes

---

## 📈 Roadmap

- [x] Week 1 — ML Foundation (TF-IDF + Random Forest + Streamlit UI)
- [ ] Week 2 — Deep Learning (CodeBERT transformer)
- [ ] Week 3 — RAG + LLM (ChromaDB + LangChain + Groq)
- [ ] Week 4 — Full Stack (FastAPI + React + Docker + Deploy)

---

## 👨‍💻 Built by

**Nitin Kumar** — building CodeSense AI as a learning project
to master AI/ML engineering from scratch.

> Day 7 of 30 — Week 1 Complete ✅