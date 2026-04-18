"""
Fake News Detector — Streamlit GUI
"""

import re
import pickle
import os
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .main-header p {
        color: #555;
        font-size: 1.05rem;
        margin-top: -0.5rem;
    }
    .result-card {
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
    }
    .result-real {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
    }
    .result-fake {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
    }
    .result-label {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .result-real .result-label { color: #155724; }
    .result-fake .result-label { color: #721c24; }
    .confidence-text {
        font-size: 1.1rem;
        margin-top: 0.3rem;
        color: #333;
    }
    .prob-bar {
        display: flex;
        border-radius: 8px;
        overflow: hidden;
        height: 32px;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .prob-real {
        background: #28a745;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .prob-fake {
        background: #dc3545;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stFileUploader"] {
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["vectorizer"], data["model"]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def predict(text, vectorizer, model):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    return {
        "label": "FAKE" if prediction == 1 else "REAL",
        "confidence": probabilities[prediction] * 100,
        "prob_real": probabilities[0] * 100,
        "prob_fake": probabilities[1] * 100,
    }


# ── UI ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📰 Fake News Detector</h1>
    <p>Paste an article or upload a text file to check if it's real or fake</p>
</div>
""", unsafe_allow_html=True)

result = load_model()

if result is None:
    st.error(
        "**Model not found.** Run `python train_model.py` first to train and save the model."
    )
    st.stop()

vectorizer, model = result

st.markdown("---")

# Input tabs
tab_paste, tab_upload = st.tabs(["Paste Text", "Upload File"])

article_text = ""

with tab_paste:
    article_text_paste = st.text_area(
        "Article text",
        height=250,
        placeholder="Paste the article text here...",
        label_visibility="collapsed",
    )

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        article_text_upload = uploaded_file.read().decode("utf-8", errors="replace")
        st.text_area(
            "File contents (preview)",
            value=article_text_upload[:2000],
            height=200,
            disabled=True,
        )

# Determine which text to use
if uploaded_file is not None:
    article_text = article_text_upload
else:
    article_text = article_text_paste

# Analyze button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("Analyze Article", use_container_width=True, type="primary")

# Results
if analyze:
    if not article_text or not article_text.strip():
        st.warning("Please paste some text or upload a file first.")
    elif len(article_text.split()) < 5:
        st.warning("Please provide a longer text for a more accurate prediction.")
    else:
        with st.spinner("Analyzing..."):
            res = predict(article_text, vectorizer, model)

        css_class = "result-real" if res["label"] == "REAL" else "result-fake"
        icon = "✅" if res["label"] == "REAL" else "🚨"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <p class="result-label">{icon} {res["label"]}</p>
            <p class="confidence-text">Confidence: {res["confidence"]:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("**Probability breakdown**")
        real_pct = res["prob_real"]
        fake_pct = res["prob_fake"]
        st.markdown(f"""
        <div class="prob-bar">
            <div class="prob-real" style="width:{real_pct}%">Real {real_pct:.1f}%</div>
            <div class="prob-fake" style="width:{fake_pct}%">Fake {fake_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Details expander
        with st.expander("Details"):
            word_count = len(article_text.split())
            st.markdown(f"- **Word count:** {word_count}")
            st.markdown(f"- **Model:** MLP Neural Network (64 neurons, 1 hidden layer)")
            st.markdown(f"- **Features:** TF-IDF (5,000 terms)")

# ── FAQ ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("### FAQ")

with st.expander("How does this app work?"):
    st.markdown(
        "You paste or upload a news article, and the app classifies it as **Real** or "
        "**Fake** using a pre-trained machine-learning model. The result includes a "
        "confidence score and a probability breakdown."
    )

with st.expander("What model is used?"):
    st.markdown(
        "The detector uses an **MLP (Multi-Layer Perceptron) neural network** with a "
        "single hidden layer of 64 neurons and ReLU activation. It was trained with "
        "scikit-learn's `MLPClassifier`."
    )

with st.expander("How is the text processed?"):
    st.markdown(
        "Each article is lowercased and stripped of punctuation/numbers. It is then "
        "converted into numerical features using **TF-IDF** (Term Frequency–Inverse "
        "Document Frequency) with up to 5,000 terms, ignoring common English stop words."
    )

with st.expander("What data was it trained on?"):
    st.markdown(
        "The model was trained on a public dataset of labeled real and fake news articles. "
        "The dataset is split 80/20 for training and evaluation."
    )

with st.expander("How accurate is it?"):
    st.markdown(
        "The model achieves high accuracy on its test set, but no model is perfect. "
        "Use the results as a helpful signal, not as a definitive verdict. Always apply "
        "your own critical thinking when evaluating news."
    )

with st.expander("What technologies power this app?"):
    st.markdown(
        "- **Streamlit** — the web interface\n"
        "- **scikit-learn** — model training and TF-IDF vectorization\n"
        "- **pandas** — data loading and preparation\n"
        "- **Python** — all application logic"
    )
