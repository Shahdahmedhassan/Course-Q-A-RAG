"""
Streamlit app - Course Q&A RAG System
Loads the model artifacts saved by the notebook (rag_model/: vector_store.faiss, chunks.pkl, config.json)
and uses them to answer student questions with source attribution.

Run:
    streamlit run app.py
"""

import os
import json
import pickle
import time
from datetime import datetime

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = "./rag_model"  # change this if your model folder lives elsewhere


# ----------------------------------------------------------------------------
# THEME / STYLING
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="Course Q&A RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#6C5CE7"      # violet
PRIMARY_DARK = "#4834D4"
ACCENT = "#00CEC9"       # teal
ACCENT_2 = "#FD79A8"     # pink
BG = "#0F1220"
CARD_BG = "#1A1E33"
CARD_BORDER = "#2E3357"
TEXT_MUTED = "#A0A3BD"

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(circle at top left, #1B1E3D 0%, {BG} 45%, #0A0C17 100%);
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #171A30 0%, #10121F 100%);
        border-right: 1px solid {CARD_BORDER};
    }}

    h1, h2, h3 {{
        color: #F1F2F9 !important;
        font-weight: 800 !important;
    }}

    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 60%, {ACCENT} 130%);
        padding: 28px 32px;
        border-radius: 18px;
        margin-bottom: 22px;
        box-shadow: 0 12px 30px rgba(108, 92, 231, 0.35);
    }}
    .hero h1 {{
        margin: 0;
        font-size: 2.1rem;
        color: white !important;
    }}
    .hero p {{
        margin: 6px 0 0 0;
        color: rgba(255,255,255,0.9);
        font-size: 1.02rem;
    }}

    .stat-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
        padding: 16px 18px;
        text-align: center;
    }}
    .stat-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {ACCENT};
    }}
    .stat-label {{
        font-size: 0.8rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    .source-card {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-left: 4px solid {ACCENT};
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }}
    .source-title {{
        font-weight: 700;
        color: #F1F2F9;
        font-size: 0.95rem;
    }}
    .source-file {{
        color: {TEXT_MUTED};
        font-size: 0.82rem;
    }}
    .score-bar-bg {{
        background: #2A2E4E;
        border-radius: 6px;
        height: 8px;
        margin-top: 8px;
        overflow: hidden;
    }}
    .score-bar-fill {{
        background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
        height: 100%;
        border-radius: 6px;
    }}

    .badge {{
        display: inline-block;
        background: rgba(108, 92, 231, 0.18);
        color: #B8AFFF;
        border: 1px solid rgba(108, 92, 231, 0.4);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }}

    div[data-testid="stChatMessage"] {{
        background: {CARD_BG};
        border: 1px solid {CARD_BORDER};
        border-radius: 14px;
    }}

    .stButton > button {{
        background: linear-gradient(120deg, {PRIMARY}, {PRIMARY_DARK});
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55em 1.4em;
    }}
    .stButton > button:hover {{
        background: linear-gradient(120deg, {ACCENT}, {PRIMARY});
        color: white;
    }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_rag_model(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    index = faiss.read_index(os.path.join(model_dir, "vector_store.faiss"))

    with open(os.path.join(model_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)

    embedding_model = SentenceTransformer(config["embedding_model_name"])
    gen_tokenizer = AutoTokenizer.from_pretrained(config["generation_model_name"])
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(config["generation_model_name"])

    return {
        "config": config,
        "index": index,
        "chunks": chunks,
        "embedding_model": embedding_model,
        "gen_tokenizer": gen_tokenizer,
        "gen_model": gen_model,
    }


def retrieve(rag, query, top_k=4, course_filter=None):
    embedding_model = rag["embedding_model"]
    index = rag["index"]
    chunks = rag["chunks"]

    query_vec = embedding_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    search_k = top_k * 3 if (course_filter and course_filter != "All courses") else top_k
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        doc = chunks[idx]
        if course_filter and course_filter != "All courses" and doc["course"] != course_filter:
            continue
        results.append({
            "score": float(score),
            "course": doc["course"],
            "source_file": doc["source_file"],
            "text": doc["text"],
        })
        if len(results) >= top_k:
            break
    return results


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {c['course']} - {c['source_file']}]\n{c['text']}" for c in retrieved_chunks
    )
    return (
        "You are a helpful teaching assistant. Using ONLY the context below, write a clear, "
        "complete answer of 2-4 full sentences that directly explains the answer to the "
        "student's question. Do not answer with a single word. If the answer is not in the "
        "context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Write your full answer below:\n"
    )


def generate_with_local_model(rag, prompt, max_new_tokens=200):
    tokenizer = rag["gen_tokenizer"]
    model = rag["gen_model"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min(60, max_new_tokens),
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        length_penalty=1.4,
        early_stopping=False,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # If generation still ends mid-sentence (no closing punctuation), trim back to the
    # last complete sentence so we never show a cut-off fragment.
    if text and text[-1] not in ".!?":
        for sep in [". ", "! ", "? "]:
            idx = text.rfind(sep)
            if idx != -1:
                text = text[: idx + 1]
                break
    return text


def generate_with_openai(prompt, api_key, model_name="gpt-4o-mini", max_tokens=300):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def answer_question(rag, query, top_k=4, course_filter=None, max_new_tokens=200,
                     engine="local", openai_api_key=None, openai_model="gpt-4o-mini"):
    retrieved = retrieve(rag, query, top_k=top_k, course_filter=course_filter)
    if not retrieved:
        return "I couldn't find anything relevant to that in the course materials.", []

    prompt = build_prompt(query, retrieved)

    if engine == "openai" and openai_api_key:
        try:
            output = generate_with_openai(prompt, openai_api_key, model_name=openai_model, max_tokens=max_new_tokens)
        except Exception as e:
            output = f"OpenAI request failed ({e}). Falling back to the local model.\n\n" + \
                      generate_with_local_model(rag, prompt, max_new_tokens=max_new_tokens)
    else:
        output = generate_with_local_model(rag, prompt, max_new_tokens=max_new_tokens)

    return output, retrieved


# ----------------------------------------------------------------------------
# APP STATE
# ----------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": ..., "sources": [...]}

if not os.path.exists(MODEL_DIR):
    st.error(
        f"Model folder not found at `{MODEL_DIR}`. Run the notebook first, then place the "
        f"`rag_model/` folder next to `app.py`."
    )
    st.stop()

with st.spinner("Loading model..."):
    rag = load_rag_model(MODEL_DIR)

courses = rag["config"]["courses"]

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    course_filter = st.selectbox("Course", ["All courses"] + courses)
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=4)
    max_new_tokens = st.slider("Max answer length (tokens)", min_value=50, max_value=400, value=200, step=10)

    st.divider()
    st.markdown("### 🧠 Answer engine")
    engine_choice = st.radio(
        "Generation model",
        ["Local (free)", "OpenAI (API key)"],
        index=0,
        label_visibility="collapsed",
    )
    engine = "local"
    openai_api_key = None
    openai_model = "gpt-4o-mini"
    if engine_choice == "OpenAI (API key)":
        engine = "openai"
        openai_api_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
        openai_model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0)
        if not openai_api_key:
            st.caption("⚠️ Enter your API key above, otherwise the local model will be used.")

    st.divider()
    st.markdown("### 📊 Knowledge base")
    st.markdown(f"""
    <div class="stat-card" style="margin-bottom:10px;">
        <div class="stat-value">{rag['config']['num_chunks']}</div>
        <div class="stat-label">Indexed chunks</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{len(courses)}</div>
        <div class="stat-label">Courses loaded</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    with st.expander("Model details"):
        st.markdown(f"**Embedding model:**\n`{rag['config']['embedding_model_name']}`")
        st.markdown(f"**Generation model:**\n`{rag['config']['generation_model_name']}`")
        st.markdown("**Courses:**")
        for c in courses:
            st.markdown(f"- {c}")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ----------------------------------------------------------------------------
# HERO HEADER
# ----------------------------------------------------------------------------

st.markdown(f"""
<div class="hero">
    <h1>📚 Course Q&A RAG System</h1>
    <p>Ask a question about your course material — answers are generated strictly from your
    documents, with full source attribution.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="stat-card"><div class="stat-value">{rag['config']['num_chunks']}</div>
                <div class="stat-label">Chunks indexed</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="stat-card"><div class="stat-value">{len(courses)}</div>
                <div class="stat-label">Courses</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-card"><div class="stat-value">{len(st.session_state.history)//2}</div>
                <div class="stat-label">Questions this session</div></div>""", unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------------
# CHAT HISTORY
# ----------------------------------------------------------------------------

for msg in st.session_state.history:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)"):
                for s in msg["sources"]:
                    pct = max(0, min(100, round(s["score"] * 100)))
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="badge">{s['course']}</span>
                        <span class="source-file">{s['source_file']}</span>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;"></div></div>
                        <div class="source-file" style="margin-top:4px;">Relevance: {pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------------------------------

query = st.chat_input("Ask a question about the course material...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving relevant material and generating answer..."):
            start = time.time()
            answer, sources = answer_question(
                rag, query, top_k=top_k, course_filter=course_filter, max_new_tokens=max_new_tokens,
                engine=engine, openai_api_key=openai_api_key, openai_model=openai_model
            )
            elapsed = time.time() - start
        st.markdown(answer)
        st.caption(f"⏱️ {elapsed:.1f}s · {len(sources)} source(s) retrieved")

        if sources:
            with st.expander(f"📎 {len(sources)} source(s)", expanded=True):
                for s in sources:
                    pct = max(0, min(100, round(s["score"] * 100)))
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="badge">{s['course']}</span>
                        <span class="source-file">{s['source_file']}</span>
                        <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%;"></div></div>
                        <div class="source-file" style="margin-top:4px;">Relevance: {pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})

# ----------------------------------------------------------------------------
# EXPORT CONVERSATION
# ----------------------------------------------------------------------------

if st.session_state.history:
    st.divider()
    transcript_lines = []
    for msg in st.session_state.history:
        role = "Student" if msg["role"] == "user" else "Assistant"
        transcript_lines.append(f"{role}: {msg['content']}")
        if msg.get("sources"):
            for s in msg["sources"]:
                transcript_lines.append(f"    Source: {s['course']} / {s['source_file']} (score={s['score']:.3f})")
    transcript = "\n\n".join(transcript_lines)

    st.download_button(
        "⬇️ Download conversation",
        data=transcript,
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=False,
    )
