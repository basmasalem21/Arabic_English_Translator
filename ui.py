import streamlit as st
from model import translate

st.set_page_config(
    page_title="Translator",
    page_icon="🌍",
    layout="centered"
)
st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #667eea 0%,
        #764ba2 50%,
        #6dd5ed 100%
    );
}

.main > div {
    padding-top: 2rem;
}

.title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:white;
    margin-bottom:20px;
}

.card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    padding: 30px;
    border-radius: 25px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

textarea {
    border-radius:15px !important;
}

.stTextArea textarea {
    background-color: rgba(255,255,255,0.9);
    color:black;
    font-size:16px;
}

.stButton button {
    width:100%;
    border:none;
    border-radius:15px;
    height:55px;
    font-size:18px;
    font-weight:bold;
    background: linear-gradient(
        90deg,
        #00c6ff,
        #0072ff
    );
    color:white;
}

.stButton button:hover {
    transform: scale(1.02);
    transition: 0.3s;
}

div[data-baseweb="select"] {
    background:white;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="title">
        🌍 AI Translator
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([5,1])

with col1:
    st.selectbox(
        "Source Language",
        ["Arabic"],
        disabled=True
    )

with col2:
    st.write("")
    st.write("")
    st.markdown("### ⇄")

st.selectbox(
    "Target Language",
    ["English"],
    disabled=True
)

input_text = st.text_area(
    "Enter text to translate",
    height=150
)

if "translation" not in st.session_state:
    st.session_state.translation = ""

col1, col2 = st.columns(2)

with col1:
    if st.button("Translate", use_container_width=True):
        if input_text.strip():
            with st.spinner("Translating..."):
                st.session_state.translation = translate(input_text)

with col2:
    if st.button("Clear", use_container_width=True):
        st.session_state.translation = ""

st.text_area(
    "Translation",
    value=st.session_state.translation,
    height=150,
    disabled=True
)

st.markdown('</div>', unsafe_allow_html=True)