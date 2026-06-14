import streamlit as st
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------

st.markdown(
    """
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #111827,
        #1e293b
    );
    color:white;
}

.main-title{
    text-align:center;
    font-size:58px;
    font-weight:800;
    margin-top:20px;
    margin-bottom:10px;
}

.sub-title{
    text-align:center;
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:40px;
}

.glass{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 24px;
    padding: 30px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
}

.result-real{
    background: rgba(34,197,94,0.15);
    border-left:5px solid #22c55e;
    border-radius:18px;
    padding:20px;
    margin-top:20px;
}

.result-fake{
    background: rgba(239,68,68,0.15);
    border-left:5px solid #ef4444;
    border-radius:18px;
    padding:20px;
    margin-top:20px;
}

.footer{
    text-align:center;
    color:#94a3b8;
    margin-top:50px;
}

textarea{
    font-size:17px !important;
}

.stButton>button{
    width:100%;
    height:60px;
    border-radius:15px;
    border:none;
    background:linear-gradient(
        135deg,
        #3b82f6,
        #8b5cf6
    );
    color:white;
    font-size:20px;
    font-weight:700;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.02);
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------- MODEL LOADING ----------------

MODEL_PATH = "models/fake_news_bert"


@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model


with st.spinner("🚀 Loading Fine-Tuned BERT Model..."):
    tokenizer, model = load_model()

# ---------------- HEADER ----------------

st.markdown(
    """
    <div class='main-title'>
        📰 Fake News Detection
    </div>
    <div class='sub-title'>
        Powered by Fine-Tuned BERT • AI News Verification System
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- MAIN CARD ----------------

st.markdown("<div class='glass'>", unsafe_allow_html=True)

news = st.text_area(
    "Paste News Article",
    height=280,
    placeholder="Paste any news article here...",
)

predict = st.button("🔍 Analyze Article")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------

if predict:
    if len(news.strip()) == 0:
        st.warning("Please enter a news article.")
        st.stop()

    inputs = tokenizer(
        news,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()

    prediction = torch.argmax(probs, dim=1).item()

    confidence = max(fake_prob, real_prob) * 100

    st.markdown("### 📊 Analysis Result")

    if prediction == 1:
        st.markdown(
            f"""
            <div class='result-real'>
                <h2>✅ REAL NEWS</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(real_prob)

    else:
        st.markdown(
            f"""
            <div class='result-fake'>
                <h2>❌ FAKE NEWS</h2>
                <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(fake_prob)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Fake Probability", f"{fake_prob * 100:.2f}%")

    with col2:
        st.metric("Real Probability", f"{real_prob * 100:.2f}%")

# ---------------- FOOTER ----------------

st.markdown(
    """
    <div class='footer'>
        <hr>
        Fake News Detection using Fine-Tuned BERT <br>
        Developed by <b>Devashish Gorai</b>
    </div>
    """,
    unsafe_allow_html=True,
)
