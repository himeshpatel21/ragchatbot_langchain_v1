import os
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv


import ragchat 


load_dotenv()


st.set_page_config(
    page_title="RAG Chat AI",
    page_icon="🤖",
    layout="wide"
)


st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stSidebar { background-color: #1A1C24; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #4A4A4A;
        color: white;
        border: none;
    }
    .stButton>button:hover { background-color: #606060; border: none; }
    .chat-bubble {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: inline-block;
    }
    .user-bubble { background-color: #262730; border-left: 5px solid #00FFAA; }
    .assistant-bubble { background-color: #161B22; border-left: 5px solid #58A6FF; }
    </style>
""", unsafe_allow_html=True)


def get_engine():
    if "engine" not in st.session_state:
        # Use keys from .env if available, otherwise from sidebar
        groq_key = st.session_state.get("groq_key") or os.getenv("GROQ_API_KEY")
        hf_token = st.session_state.get("hf_token") or os.getenv("HF_TOKEN")
        
        if not groq_key:
            st.sidebar.error("⚠️ Groq API Key Missing!")
            st.stop()
            
        st.session_state.engine = ragchat.ragchat(groq_api_key=groq_key, hf_token=hf_token)
    return st.session_state.engine

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.title("⚙️ Control Panel")
    
    with st.expander("API Credentials"):
        groq_input = st.text_input("Groq Key", value=os.getenv("GROQ_API_KEY", ""), type="password")
        hf_input = st.text_input("HF Token", value=os.getenv("HF_TOKEN", ""), type="password")
        if st.button("Update Keys"):
            st.session_state.groq_key = groq_input
            st.session_state.hf_token = hf_input
            if "engine" in st.session_state: del st.session_state["engine"]
            st.rerun()

    st.divider()
    st.subheader("Add Knowledge")
    tabs = st.tabs(["🌐 URL", "📄 PDF", "✍️ Text", "📺 YT", "📁 TXT"])

    # URL
    with tabs[0]:
        url_in = st.text_input("Web URL")
        if st.button("Ingest URL") and url_in:
            docs = ragchat.load_from_url(url_in)
            get_engine().add_source(docs)
            st.success("URL Loaded!")

    # PDF
    with tabs[1]:
        pdf_in = st.file_uploader("Upload PDF", type="pdf")
        if st.button("Ingest PDF") and pdf_in:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_in.read())
                docs = ragchat.load_from_pdf(tmp.name, pdf_in.name)
                get_engine().add_source(docs)
            st.success("PDF Loaded!")

    # Manual Text
    with tabs[2]:
        raw_text = st.text_area("Paste content here")
        if st.button("Ingest Text") and raw_text:
            docs = ragchat.load_from_text(raw_text)
            get_engine().add_source(docs)
            st.success("Text Loaded!")

    # YouTube
    with tabs[3]:
        yt_url = st.text_input("YouTube Link")
        if st.button("Ingest Video") and yt_url:
            docs = ragchat.load_from_youtube(yt_url)
            get_engine().add_source(docs)
            st.success("Transcript Loaded!")

    # TXT File
    with tabs[4]:
        txt_file = st.file_uploader("Text File", type="txt")
        if st.button("Ingest TXT") and txt_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(txt_file.read())
                # Using the specific name from your ragchat.py file
                docs = ragchat.load_from_text_file(tmp.name, txt_file.name)
                # Check if docs is None (fixing the missing return in your original file)
                if docs:
                    get_engine().add_source(docs)
                    st.success("File Loaded!")
                else:
                    st.error("Error: load_from_text_file returned nothing. Check your engine code.")

# Main Chat UI
st.title("💬 RAG Intelligence")

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        engine = get_engine()
        try:
            result = engine.chat(prompt)
            answer = result["answer"]
            st.write(answer)
            
            if result["sources"]:
                with st.expander("📚 View Sources"):
                    for s in result["sources"]:
                        st.markdown(f"**{s['name']}** ({s['type']})")
                        if s.get('snippet'): st.caption(f"Snippet: {s['snippet']}...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")