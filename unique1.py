import os
import io
import csv
import json
import time
import random
import re
import traceback
from typing import List, Dict, Tuple, Optional
import threading

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from streamlit_lottie import st_lottie
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from dotenv import load_dotenv
from googleapiclient.discovery import build
import google.generativeai as genai

# Optional TTS
try:
    from gtts import gTTS
except Exception:
    gTTS = None

# =====================================================
# ENVIRONMENT SETUP
# =====================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="📚 StudyMate - Enhanced AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD LOTTIE ANIMATION
# =====================================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

animation_url = "https://assets5.lottiefiles.com/packages/lf20_j1adxtyb.json"
lottie_animation = load_lottieurl(animation_url)

# =====================================================
# MODEL CACHING
# =====================================================
@st.cache_resource
def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)

@st.cache_resource
def load_qa_pipeline():
    try:
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model
    except Exception:
        return None, None

@st.cache_resource
def get_summarization_pipeline():
    try:
        pipe = pipeline("summarization", model="facebook/bart-large-cnn")
        return pipe
    except Exception:
        return None

# Load models
tokenizer, model = load_qa_pipeline()
summarizer = get_summarization_pipeline()

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def ensure_session_state():
    st.session_state.setdefault('studymate', None)
    st.session_state.setdefault('documents_processed', False)
    st.session_state.setdefault('chat_history', [])
    st.session_state.setdefault('questions_asked', 0)
    st.session_state.setdefault('flashcards', [])
    st.session_state.setdefault('quizzes', [])
    st.session_state.setdefault('processing_done', False)
    st.session_state.setdefault('error', None)
    st.session_state.setdefault('elapsed_time', 0)

ensure_session_state()

def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def tts_bytes(text: str) -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        fp = io.BytesIO()
        gTTS(text).write_to_fp(fp)
        return fp.getvalue()
    except Exception:
        return None

def save_uploaded_file(uploaded_file):
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return file_path

# =====================================================
# YOUTUBE API HELPER
# =====================================================
def get_youtube_videos(query):
    """Fetch top 3 related YouTube videos."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            part="snippet", q=query, type="video", maxResults=3
        )
        response = request.execute()

        videos = []
        for item in response.get("items", []):
            videos.append({
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"]
            })
        return videos
    except Exception as e:
        return []

# =====================================================
# CORE STUDYMATE CLASS
# =====================================================
class StudyMate:
    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.index: Optional[faiss.IndexFlatL2] = None
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []

    def extract_text_from_pdf(self, pdf_file) -> List[Dict]:
        """Extract text from PDF file"""
        data = pdf_file.getvalue()
        doc = fitz.open(stream=data, filetype="pdf")
        text_chunks: List[Dict] = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
                for para in paragraphs:
                    if len(para) > 60:
                        text_chunks.append({
                            "text": para,
                            "source": getattr(pdf_file, "name", "uploaded.pdf"),
                            "page": page_num + 1
                        })
        doc.close()
        return text_chunks

    def create_embeddings_and_index(self, text_chunks: List[Dict]) -> int:
        """Create embeddings and build FAISS index"""
        if not text_chunks:
            return 0

        texts = [c["text"] for c in text_chunks]
        with st.spinner("🔍 Creating embeddings..."):
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )

        embeddings = embeddings.astype("float32", copy=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.chunks = texts
        self.chunk_metadata = text_chunks
        return len(texts)

    def search_similar_chunks(self, query: str, k: int = 3) -> List[Tuple[str, Dict]]:
        """Search for similar chunks using FAISS"""
        if self.embedding_model is None or self.index is None or not self.chunks:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32", copy=False)

        k = max(1, min(k, len(self.chunks)))
        distances, indices = self.index.search(query_embedding, k)

        results: List[Tuple[str, Dict]] = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], self.chunk_metadata[idx]))
        return results

    def generate_answer_with_gemini(self, query: str, context_chunks: List[str]) -> str:
        """Use Gemini API to generate answers"""
        if not GEMINI_API_KEY:
            return "⚠ Gemini API key not configured. Using alternative method..."
        
        try:
            context = "\n".join(context_chunks[:5])
            prompt = f"""You are StudyMate AI Assistant.
Use the following study material to answer the question.
If the answer is not in the content, say "I don't have enough information."

Context:
{context}

Question:
{query}

Answer:"""
            
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"⚠ Error with Gemini: {str(e)}"

    def generate_answer_with_gpt(self, query: str, context_chunks: List[str]) -> str:
        """Use local GPT model to generate answers"""
        if not model or not tokenizer:
            return "❌ Language model not available."

        context = " ".join(context_chunks[:2])[:900]
        if not context.strip():
            return "❌ I couldn't find relevant information in your documents to answer that question."

        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        try:
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            return answer.strip() or "❌ Unable to generate a complete response."
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def generate_answer_with_summarizer(self, query: str, context_chunks: List[str]) -> str:
        """Use BART summarizer to generate answers"""
        if not summarizer:
            return "❌ Summarization model not available."

        context = " ".join(context_chunks[:4])
        if not context.strip():
            return "❌ I couldn't find relevant information in your documents to answer that question."

        try:
            text_to_summarize = f"Question about: {query}. Content: {context}"
            if len(text_to_summarize) > 1024:
                text_to_summarize = text_to_summarize[:1024]
            summary = summarizer(text_to_summarize, max_length=200, min_length=40)
            return summary[0].get('summary_text', '').strip()
        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"

    def extract_key_info(self, query: str, context_chunks: List[str]) -> str:
        """Extract key information using keyword matching"""
        context = " ".join(context_chunks[:4]).lower()
        query_lower = query.lower()
        if not context.strip():
            return "❌ I couldn't find relevant information in your documents to answer that question."

        sentences = re.split(r'[.!?]+', context)
        query_words = set(query_lower.split())
        relevant_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(query_words.intersection(sentence_words)) > 0:
                relevant_sentences.append(sentence.strip())
        if relevant_sentences:
            return "Based on your documents:\n• " + "\n• ".join(relevant_sentences[:3])
        else:
            return "❌ No directly relevant information found for your specific question."

    def heuristic_flashcards(self, chunks: List[str], n: int = 10) -> List[Tuple[str, str]]:
        """Generate flashcards from document chunks"""
        text = " ".join(chunks)
        sents = re.split(r'(?<=[.!?]) +', text)
        sents = [s.strip() for s in sents if len(s.strip()) > 40]
        sents = sorted(sents, key=lambda s: -len(s))
        cards = []
        for s in sents:
            words = re.findall(r"\w+", s)
            long_words = [w for w in words if len(w) > 6]
            if not long_words:
                continue
            target = long_words[0]
            q = s.replace(target, "_____")
            a = target
            cards.append((clean_text(q), clean_text(a)))
            if len(cards) >= n:
                break
        return cards

    def generate_quiz(self, chunks: List[str], n: int = 5) -> List[Dict]:
        """Generate multiple choice quiz from chunks"""
        quiz = []
        sents = []
        for c in chunks:
            sents.extend(re.split(r'(?<=[.!?]) +', c))
        sents = [s.strip() for s in sents if len(s.strip()) > 40]
        random.shuffle(sents)
        used = 0
        for s in sents:
            words = re.findall(r"\b[A-Za-z]{5,}\b", s)
            if len(words) < 1:
                continue
            answer = random.choice(words)
            distractors = set()
            all_words = re.findall(r"\b[A-Za-z]{5,}\b", " ".join(chunks))
            random.shuffle(all_words)
            for w in all_words:
                if w.lower() != answer.lower() and len(distractors) < 3:
                    distractors.add(w)
            if len(distractors) < 3:
                continue
            options = list(distractors) + [answer]
            random.shuffle(options)
            quiz.append({
                'question': s.replace(answer, '_____'),
                'answer': answer,
                'options': options
            })
            used += 1
            if used >= n:
                break
        return quiz

# =====================================================
# UI STYLING
# =====================================================
def inject_dark_mode_css(enabled: bool):
    if not enabled:
        return
    st.markdown("""
    <style>
    .stApp { background: #0b0f1a; color: #e6eef8; }
    .stButton>button { background-color:#2563eb; color:white; border-radius: 8px; }
    .stButton>button:hover { background-color:#1d4ed8; }
    .css-1d391kg { background-color: #0b0f1a; }
    .stTextInput>div>div>input { background:#0b1220; color:#e6eef8; border-radius: 6px; }
    .stSelectbox>div>div>select { background:#0b1220; color:#e6eef8; }
    .stExpander { border-color: #2563eb; }
    .stMetric { background-color: #1a1f3a; border-radius: 8px; padding: 15px; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# MAIN APP
# =====================================================
def main():
    st.sidebar.title("⚙️ StudyMate Controls")
    
    # Dark mode toggle
    dark = st.sidebar.checkbox("🌙 Dark Mode", value=True)
    inject_dark_mode_css(dark)

    st.title("📚 StudyMate — Enhanced AI Study Assistant")
    st.write("Powered by Gemini, local AI models, FAISS embeddings, flashcards, quizzes, and more!")

    if lottie_animation:
        st_lottie(lottie_animation, height=150)

    # =========== SIDEBAR SETUP ===========
    with st.sidebar:
        st.header("📄 Upload & Process")
        uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
        
        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} file(s) ready to process")
            if st.button("🔄 Process Documents", use_container_width=True):
                all_chunks = []
                if st.session_state.get('studymate') is None:
                    st.session_state['studymate'] = StudyMate()
                
                progress_bar = st.progress(0)
                for idx, f in enumerate(uploaded_files):
                    try:
                        c = st.session_state['studymate'].extract_text_from_pdf(f)
                        all_chunks.extend(c)
                        progress_bar.progress((idx + 1) / len(uploaded_files))
                    except Exception as e:
                        st.error(f"Failed to read {getattr(f,'name','file')}: {e}")
                
                if all_chunks:
                    cnt = st.session_state['studymate'].create_embeddings_and_index(all_chunks)
                    st.session_state['documents_processed'] = True
                    st.session_state['processing_done'] = True
                    st.success(f"✅ Processed {cnt} chunks from {len(uploaded_files)} file(s)")
                    
                    # Auto-generate flashcards
                    fc = st.session_state['studymate'].heuristic_flashcards(
                        [t for t, _ in [(ch['text'], ch) for ch in all_chunks]], n=12
                    )
                    st.session_state['flashcards'] = fc
                    st.session_state['questions_asked'] = 0
                else:
                    st.error("❌ No readable text found.")

        st.markdown("---")
        st.header("🎯 Settings")
        ai_method = st.selectbox(
            "Answer Method",
            ["Keyword Extraction (Fast)", "GPT Generation", "Summarization (BART)", "Gemini (Cloud)"],
            index=0
        )
        
        k = st.slider("Context chunks to use", 1, 8, 4)
        
        st.markdown("---")
        st.header("🔊 Features")
        tts_enabled = st.checkbox("Enable Text-to-Speech", value=(gTTS is not None))
        show_youtube = st.checkbox("Show YouTube Videos", value=(YOUTUBE_API_KEY is not None))

    # =========== MAIN CONTENT ===========
    col1, col2 = st.columns([2.5, 1.5])

    with col1:
        st.header("💬 Ask Questions")
        
        if not st.session_state.get('documents_processed'):
            st.info("📌 Upload and process PDF documents using the sidebar to get started!")
        else:
            user_question = st.text_input(
                "What would you like to know?",
                key='question_input',
                placeholder="e.g., What are the main concepts explained in this document?"
            )
            
            if st.button("🔎 Ask StudyMate", use_container_width=True) and user_question.strip():
                st.session_state['questions_asked'] += 1
                sm: StudyMate = st.session_state['studymate']
                
                results = sm.search_similar_chunks(user_question, k=k)
                context_texts = [t for t, _ in results]
                
                # Select answer method
                if ai_method == "Keyword Extraction (Fast)":
                    answer = sm.extract_key_info(user_question, context_texts)
                    method_used = 'keyword'
                elif ai_method == "GPT Generation":
                    answer = sm.generate_answer_with_gpt(user_question, context_texts)
                    method_used = 'gpt'
                elif ai_method == "Summarization (BART)":
                    answer = sm.generate_answer_with_summarizer(user_question, context_texts)
                    method_used = 'summarizer'
                else:  # Gemini
                    answer = sm.generate_answer_with_gemini(user_question, context_texts)
                    method_used = 'gemini'
                
                # Store in chat history
                st.session_state['chat_history'].append({
                    'q': user_question,
                    'a': answer,
                    'method': method_used
                })
                
                # Display answer
                st.markdown("### 🎯 Answer")
                st.markdown(answer)
                
                # TTS
                if tts_enabled and gTTS is not None:
                    audio = tts_bytes(answer)
                    if audio:
                        st.audio(audio, format='audio/mp3')
                
                # YouTube recommendations
                if show_youtube and YOUTUBE_API_KEY:
                    st.markdown("---")
                    st.markdown("### 🎥 Recommended YouTube Videos")
                    videos = get_youtube_videos(user_question)
                    if videos:
                        cols = st.columns(3)
                        for col, video in zip(cols, videos):
                            with col:
                                st.image(video['thumbnail'])
                                st.markdown(f"[{video['title']}]({video['url']})", unsafe_allow_html=True)
                    else:
                        st.info("No related videos found.")
                
                # Show sources
                with st.expander("📖 Sources & Snippets"):
                    if results:
                        for i, (text, meta) in enumerate(results[:5]):
                            st.markdown(f"**Source {i+1}:** {meta['source']} (Page {meta['page']})")
                            snippet = text if len(text) < 300 else text[:300] + '...'
                            st.text(snippet)
                            st.markdown('---')
                    else:
                        st.write("No source snippets found.")

    with col2:
        st.header("📊 Session Stats")
        st.metric("❓ Questions Asked", st.session_state.get('questions_asked', 0))
        st.metric("🃏 Flashcards", len(st.session_state.get('flashcards', [])))
        st.metric("📝 Chat History", len(st.session_state.get('chat_history', [])))

        st.markdown("---")

        # ========== FLASHCARDS ==========
        with st.expander("🃏 Flashcards (Auto-generated)"):
            fcards = st.session_state.get('flashcards', [])
            if not fcards:
                st.write("No flashcards available. Process documents first.")
            else:
                idx = st.number_input("Flashcard #", min_value=1, max_value=len(fcards), value=1)
                q, a = fcards[int(idx) - 1]
                st.markdown(f"**Q:** {q}")
                if st.button("Show answer", key=f"show_ans_{idx}", use_container_width=True):
                    st.markdown(f"**A:** {a}")
                
                # Download flashcards
                if st.button("📥 Download as CSV", use_container_width=True):
                    buf = io.StringIO()
                    w = csv.writer(buf)
                    w.writerow(["Question", "Answer"])
                    for qq, aa in fcards:
                        w.writerow([qq, aa])
                    st.download_button(
                        "Download Flashcards CSV",
                        data=buf.getvalue(),
                        file_name="studymate_flashcards.csv",
                        use_container_width=True
                    )

        st.markdown("---")

        # ========== QUIZ MODE ==========
        with st.expander("🧩 Quiz Mode"):
            if st.button("Generate New Quiz", use_container_width=True):
                if st.session_state.get('studymate'):
                    sm: StudyMate = st.session_state['studymate']
                    quiz = sm.generate_quiz(sm.chunks, n=5)
                    st.session_state['quizzes'] = quiz

            quiz = st.session_state.get('quizzes', [])
            if quiz:
                score = 0
                for i, qdata in enumerate(quiz):
                    st.markdown(f"**Q{i+1}.** {qdata['question']}")
                    choice = st.radio(f"Select answer for Q{i+1}", qdata['options'], key=f"quiz_{i}")
                    if st.button(f"Submit Q{i+1}", key=f"submit_{i}", use_container_width=True):
                        if choice == qdata['answer']:
                            st.success("✅ Correct!")
                            score += 1
                        else:
                            st.error(f"❌ Incorrect — Correct answer: **{qdata['answer']}**")
                if len(quiz) > 0:
                    st.info(f"📊 Quiz completed. Score: {score}/{len(quiz)}")

        st.markdown("---")

        # ========== CHAT HISTORY ==========
        with st.expander("💬 Chat History"):
            history = st.session_state.get('chat_history', [])
            if not history:
                st.write("No interactions yet.")
            else:
                for entry in reversed(history[-10:]):
                    st.markdown(f"**Q:** {entry['q']}")
                    st.markdown(f"**A:** {entry['a']}")
                    st.caption(f"_Method: {entry['method']}_")
                    st.markdown('---')

    st.markdown("---")
    st.markdown("🎓 **Made with ❤️ — StudyMate Enhanced** | Powered by Gemini, FAISS, Transformers & Streamlit")

if __name__ == '__main__':
    main()
