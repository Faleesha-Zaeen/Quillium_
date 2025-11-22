import streamlit as st
import fitz  # PyMuPDF
import random, re, os, time
import requests, json
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; environment variables will still be read from the environment
    pass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from nltk.corpus import wordnet
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Download WordNet for synonyms
nltk.download('wordnet')

st.set_page_config(page_title="Quillium", page_icon="🌿", layout="wide")

# ---------------- Cached Translator ----------------
@st.cache_resource(show_spinner=False)
def load_translator():
    try:
        model_name = "Helsinki-NLP/opus-mt-en-mul"
        local_dir = "./models/opus-mt-en-mul"
        os.makedirs(local_dir, exist_ok=True)

        if os.path.exists(os.path.join(local_dir, "config.json")):
            tok = AutoTokenizer.from_pretrained(local_dir)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(local_dir)
        else:
            tok = AutoTokenizer.from_pretrained(model_name)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            tok.save_pretrained(local_dir)
            mdl.save_pretrained(local_dir)
        return tok, mdl
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

# Translator will be loaded lazily when translation is first requested
# This avoids long model downloads during app startup which can cause timeouts.
if 'translator_tokenizer' not in st.session_state:
    st.session_state.translator_tokenizer = None
    st.session_state.translator_model = None


def translate_text(text, target_lang):
    if target_lang == "English" or not text.strip():
        return text

    # Lazy load model
    if ('translator_tokenizer' not in st.session_state) or (st.session_state.translator_tokenizer is None):
        with st.spinner("🔄 Loading translation model... (one time only)"):
            tok, mdl = load_translator()
            st.session_state.translator_tokenizer = tok
            st.session_state.translator_model = mdl

    prefix_map = {
        "Spanish": ">>spa<< ",
        "French": ">>fra<< ",
        "German": ">>deu<< ",
        "Italian": ">>ita<< ",
        "Portuguese": ">>por<< ",
        "Russian": ">>rus<< ",
        "Dutch": ">>nld<< ",
        "Polish": ">>pol<< ",
        "Ukrainian": ">>ukr<< ",
        "Romanian": ">>ron<< ",
        "Greek": ">>ell<< ",
        "Czech": ">>ces<< ",
        "Swedish": ">>swe<< ",
        "Norwegian": ">>nor<< ",
        "Danish": ">>dan<< ",
        "Finnish": ">>fin<< ",
        "Hungarian": ">>hun<< ",
        "Bulgarian": ">>bul<< ",
        "Chinese": ">>zho<< ",
        "Japanese": ">>jpn<< ",
        "Korean": ">>kor<< ",
        "Arabic": ">>ara<< ",
        "Turkish": ">>tur<< ",
        "Thai": ">>tha<< ",
        "Vietnamese": ">>vie<< ",
        "Indonesian": ">>ind<< ",
        "Malay": ">>msa<< ",
        "Filipino": ">>tgl<< ",
        "Persian": ">>fas<< ",
        "Hindi": ">>hin<< ",
        "Tamil": ">>tam<< ",
        "Telugu": ">>tel<< ",
        "Kannada": ">>kan<< ",
        "Malayalam": ">>mal<< ",
        "Bengali": ">>ben<< ",
        "Marathi": ">>mar<< ",
        "Gujarati": ">>guj<< ",
        "Punjabi": ">>pan<< ",
        "Urdu": ">>urd<< ",
    }

    prefix = prefix_map.get(target_lang, "")

    formatted = f"{prefix}Rewrite clearly and naturally for a student: {text}"

    try:
        inputs = st.session_state.translator_tokenizer(formatted, return_tensors="pt", truncation=True)
        outputs = st.session_state.translator_model.generate(**inputs, max_new_tokens=180)
        return st.session_state.translator_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        return text + " [⚠ Translation Failed]"


# ---------------- Cached PDF Processing ----------------
@st.cache_data(show_spinner="📄 Processing PDF...", ttl=3600)
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text("text").strip()
            if page_text:
                full_text += page_text + " "

        page_count = doc.page_count
        doc.close()

        if len(full_text.strip()) < 50:
            return (
                "This document contains minimal text. Please try a document with more content.",
                page_count,
            )

        return full_text.strip(), page_count
    except Exception as e:
        return f"Error processing PDF: {e}", 0


@st.cache_data(show_spinner=False, ttl=1800)
def extract_questions_from_text(text):
    if text.startswith("Error") or text.startswith("This document contains"):
        return []

    questions = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = sentences[:200]  # up to 200 sentences

    question_patterns = [
        # Definition patterns
        (r"^(.*?)\s+is\s+(?:an?\s+)?(.*)", "What is {}?"),
        (r"^(.*?)\s+are\s+(.*)", "What are {}?"),
        (r"^(.*?)\s+means\s+(.*)", "What does {} mean?"),
        (r"^(.*?)\s+refers to\s+(.*)", "What does {} refer to?"),
        (r"^(.*?)\s+can be defined as\s+(.*)", "How is {} defined?"),
        (r"^(.*?)\s+is defined as\s+(.*)", "How is {} defined?"),
        (r"^(.*?)\s+known as\s+(.*)", "What is {} known as?"),
        (r"^(.*?)\s+called\s+(.*)", "What is {} called?"),
        # Process/Description patterns
        (r"^(.*?)\s+involves\s+(.*)", "What does {} involve?"),
        (r"^(.*?)\s+includes\s+(.*)", "What does {} include?"),
        (r"^(.*?)\s+consists of\s+(.*)", "What does {} consist of?"),
        (r"^(.*?)\s+occurs when\s+(.*)", "When does {} occur?"),
        (r"^(.*?)\s+happens when\s+(.*)", "When does {} happen?"),
        # Purpose/Function patterns
        (r"^(.*?)\s+is used to\s+(.*)", "What is {} used for?"),
        (r"^(.*?)\s+is used for\s+(.*)", "What is {} used for?"),
        (r"^(.*?)\s+helps to\s+(.*)", "How does {} help?"),
        (r"^(.*?)\s+allows\s+(.*)", "What does {} allow?"),
        # Characteristic patterns
        (r"^(.*?)\s+has\s+(.*)", "What does {} have?"),
        (r"^(.*?)\s+contains\s+(.*)", "What does {} contain?"),
        (r"^(.*?)\s+provides\s+(.*)", "What does {} provide?"),
    ]

    used_sentences = set()

    for sent in sentences:
        if len(questions) >= 20:
            break

        sent = sent.strip()
        if len(sent.split()) < 5 or len(sent) < 20:
            continue

        if sent in used_sentences:
            continue

        for pattern, question_template in question_patterns:
            match = re.match(pattern, sent, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                answer = match.group(2).strip()

                subject = re.sub(r"[.,;:]$", "", subject)
                answer = re.sub(r"[.,;:]$", "", answer)

                if (
                    1 <= len(subject.split()) <= 8
                    and len(answer.split()) >= 2
                    and len(answer) > 10
                ):
                    question_text = question_template.format(subject)
                    questions.append(
                        {
                            "question": question_text,
                            "answer": answer,
                            "source_sentence": sent,
                        }
                    )
                    used_sentences.add(sent)
                    break

    # Fallback
    if len(questions) < 20:
        remaining_slots = 20 - len(questions)
        important_sentences = [
            s
            for s in sentences
            if s not in used_sentences
            and len(s.split()) > 8
            and len(s) > 30
        ][: remaining_slots * 2]

        question_types = [
            "What is the main idea?",
            "What key concept is described?",
            "What important point is made?",
            "What is being explained?",
            "What does this describe?",
            "What process is outlined?",
            "What principle is discussed?",
        ]

        for i, sent in enumerate(important_sentences[:remaining_slots]):
            if len(questions) >= 20:
                break
            q_type = question_types[i % len(question_types)]
            questions.append(
                {"question": q_type, "answer": sent, "source_sentence": sent}
            )

    return questions[:20]


# ---------------- Smart Distractors (fallback) ----------------
def generate_smart_distractors(answer, n=3):
    if not answer or len(answer.split()) == 0:
        return ["Not correct", "Incorrect choice", "Different concept"]

    words = answer.split()
    distractors = set()

    attempts = min(20, len(words) * 2)
    for _ in range(attempts):
        if len(distractors) >= n:
            break

        new_words = words.copy()
        idx = random.randint(0, len(words) - 1)
        word = words[idx]

        if len(word) <= 2:
            continue

        try:
            syns = wordnet.synsets(word)
            if syns:
                lemmas = []
                for syn in syns[:2]:
                    for lemma in syn.lemmas()[:2]:
                        lemma_name = lemma.name().replace("_", " ")
                        if (
                            lemma_name.lower() != word.lower()
                            and len(lemma_name.split()) == 1
                            and len(lemma_name) > 2
                        ):
                            lemmas.append(lemma_name)

                if lemmas:
                    new_words[idx] = random.choice(lemmas)
                    fake = " ".join(new_words)
                    if 10 < len(fake) < 100:
                        distractors.add(fake)
        except Exception:
            pass

    filler_groups = [
        [
            "A different concept",
            "Another related idea",
            "A separate topic",
        ],
        [
            "Alternative explanation",
            "Different interpretation",
            "Another possible description",
        ],
        [
            "Other framework",
            "Other approach",
            "Other principle",
        ],
        [
            "Not the correct concept",
            "Not the right answer",
            "Incorrect understanding",
        ],
    ]

    while len(distractors) < n:
        group = random.choice(filler_groups)
        distractors.update(group)

    return list(distractors)[:n]


# --------- Shared MCQ validation & cleaning ----------


def validate_and_clean_mcq(item):
    """Ensure each MCQ has clear question, correct answer, 3 good distractors, difficulty label, and 4 unique options."""
    question = (item.get("question") or "").strip()
    answer = (item.get("answer") or "").strip()

    if not question or not answer:
        return None

    # Drop absurdly short or huge stuff
    if len(question.split()) < 4 or len(question) > 220:
        return None
    if len(answer.split()) < 1 or len(answer) > 200:
        return None

    distractors = item.get("distractors") or item.get("incorrect_options") or item.get(
        "options"
    )
    if distractors is None:
        distractors = []

    # Normalize to strings
    distractors = [str(d).strip() for d in distractors if str(d).strip()]

    # Remove exact duplicates and the true answer from distractors
    filtered = []
    seen = set()
    for d in distractors:
        if d.lower() == answer.lower():
            continue
        if d.lower() in seen:
            continue
        seen.add(d.lower())
        filtered.append(d)
    distractors = filtered

    # If less than 3, generate fallback distractors from answer
    if len(distractors) < 3:
        needed = 3 - len(distractors)
        extra = generate_smart_distractors(answer, n=needed)
        distractors.extend(extra)

    distractors = distractors[:3]

    # Build final options: 3 distractors + 1 correct answer
    options = distractors + [answer]
    options = [o for o in options if o.strip()]
    # Deduplicate options again
    seen_opts = set()
    final_options = []
    for o in options:
        low = o.lower()
        if low in seen_opts:
            continue
        seen_opts.add(low)
        final_options.append(o)

    # If we somehow lost options, pad with simple fallbacks (rare)
    while len(final_options) < 4:
        final_options.append(f"Option {len(final_options)+1}")

    # Shuffle options
    random.shuffle(final_options)

    # Difficulty
    difficulty = (item.get("difficulty") or "").strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        difficulty = "medium"

    return {
        "question": question,
        "answer": answer,
        "options": final_options,
        "difficulty": difficulty,
    }

def fallback_mcqs(text, count=10):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s for s in sentences if len(s.split()) > 5]

    if not sentences:
        sentences = ["This text is too short to extract structured meaning."]

    mcqs = []
    for i in range(min(count, len(sentences))):
        s = sentences[i][:120] + "..."
        mcqs.append({
            "question": f"What is the main idea of: '{s}'?",
            "answer": sentences[i],
            "options": [
                sentences[i],
                "Unrelated concept",
                "Incorrect interpretation",
                "Cannot be determined"
            ],
            "difficulty": "easy"
        })
    return mcqs


def translate_mcqs(mcqs, target_lang):
    if target_lang == "English":
        return mcqs

    translated = []
    for q in mcqs:
        new_q = {
            "question": translate_text(q["question"], target_lang),
            "answer": translate_text(q["answer"], target_lang),
            "options": list({translate_text(opt, target_lang) for opt in q["options"]}),
            "difficulty": q.get("difficulty", "medium"),
        }

        # Ensure exactly 4 unique options
        while len(new_q["options"]) < 4:
            new_q["options"].append("Option " + str(len(new_q["options"]) + 1))

        random.shuffle(new_q["options"])
        translated.append(new_q)

    return translated


# ---------------- IMPROVED MCQ GENERATOR (💚 FINAL WORKING VERSION) ----------------
@st.cache_data(show_spinner="🤖 Generating questions...", ttl=1800)
def make_mcqs(text, lang="English", max_questions=20):

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        st.warning("⚠️ OPENROUTER_API_KEY not set — using fallback generator.")
        return fallback_mcqs(text, max_questions)

    # 🔹 Clean and shorten text for better prompt quality
    if len(text) > 6000:
        text = text[:6000]

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://quillium.app",  
        "X-Title": "Quillium AI MCQ Generator"
    }

    prompt = f"""
You are an educational content generator.

Create EXACTLY {max_questions} multiple choice questions from the text.

📌 Required output format (strict JSON):

[
  {{
    "question": "...",
    "answer": "...",
    "options": ["..","..","..",".."],
    "difficulty": "easy | medium | hard"
  }}
]

Rules:
- Each question MUST have exactly 4 options.
- One and only one correct answer.
- Avoid vague or generic questions.
- Use clear language appropriate for students.
- Do NOT include explanations, notes, or extra text.
- If the document is not educational (resume, letter, etc.) generate general comprehension questions.

TEXT:
{text}
"""

    body = {
        "model": "deepseek/deepseek-r1:7b",
        "messages": [
            {"role": "system", "content": "Respond ONLY in JSON. No extra text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.25,
        "max_tokens": 2200
    }

    # ---------- SEND REQUEST ----------
    try:
        response = requests.post(url, headers=headers, json=body, timeout=80)
        response.raise_for_status()
        raw_output = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"❌ AI request failed: {e}")
        return fallback_mcqs(text, max_questions)

    # ---------- CLEAN OUTPUT ----------
    clean = (
        raw_output.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    # ---------- PARSE OR FIX JSON ----------
    try:
        data = json.loads(clean)
    except:
        # AI sometimes adds trailing commas → fix it automatically
        clean = re.sub(r",\s*([\]}])", r"\1", clean)
        clean = clean.replace("\n", "")
        try:
            data = json.loads(clean)
        except:
            st.warning("⚠️ JSON unreadable — generating fallback questions.")
            return fallback_mcqs(text, max_questions)

    # ---------- VALIDATE QUESTIONS ----------
    final = []
    for item in data:
        fixed = validate_and_clean_mcq(item)
        if fixed:
            final.append(fixed)

    # If usable questions still < requested → fill missing with rule-based
    if len(final) < max_questions:
        final.extend(fallback_mcqs(text, max_questions - len(final)))

    return final[:max_questions]


def translate_full_json(mcqs, lang):
    if lang == "English":
        return mcqs

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return mcqs

    prompt = f"""
Translate the following JSON into **{lang}**.
Keep the JSON structure EXACTLY the same.
Do NOT add explanations or modify the format.

JSON:
{json.dumps(mcqs, ensure_ascii=False)}
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "deepseek/deepseek-r1:7b",
        "messages": [
            {"role": "system", "content": "Return STRICT JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2200,
    }

    try:
        r = requests.post(url, headers=headers, json=body, timeout=60)
        result = r.json()["choices"][0]["message"]["content"]
        result = result.replace("```json", "").replace("```", "").strip()
        return json.loads(result)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return mcqs


def make_flashcards(text, lang="English", max_cards=20):
    mcqs = make_mcqs(text, lang="English", max_questions=max_cards)

    # Use only question + answer
    flashcards = [{"question": q["question"], "answer": q["answer"]} for q in mcqs]

    # Translate full flashcard JSON if needed
    if lang != "English":
        flashcards = translate_full_json(flashcards, lang)

    return flashcards


def init_progress():
    if "quiz_progress" not in st.session_state:
        st.session_state.quiz_progress = {
            "total_questions": 0,
            "correct_answers": 0,
            "incorrect_answers": 0,
            "quizzes_taken": 0,
            "flashcards_studied": 0,
        }


def update_progress(correct=False, flashcard_studied=False):
    init_progress()
    if correct is not None:
        st.session_state.quiz_progress["total_questions"] += 1
        if correct:
            st.session_state.quiz_progress["correct_answers"] += 1
        else:
            st.session_state.quiz_progress["incorrect_answers"] += 1
    if flashcard_studied:
        st.session_state.quiz_progress["flashcards_studied"] += 1


# ---------------- Main App ----------------
def main():
    st.markdown(
        """
    <h1 style='color: #38ef7d; text-align: center; font-size: 3em; margin-bottom: 0;'>🌿Quillium</h1>
    <p style='color: #666; text-align: center; font-size: 1.2em; margin-top: 0;'>Quiz. Learn. Conquer.</p>
    """,
        unsafe_allow_html=True,
    )

    init_progress()
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Quiz"
    if "flashcard_index" not in st.session_state:
        st.session_state.flashcard_index = 0
    if "show_answer" not in st.session_state:
        st.session_state.show_answer = False
    if "processed_file_id" not in st.session_state:
        st.session_state.processed_file_id = None

    # Sidebar
    st.sidebar.markdown(
        """
    <div style='background: #1a1a1a; padding: 20px; border-radius: 10px; border-left: 4px solid #38ef7d; color: white;'>
        <h2 style='color: #38ef7d; margin-bottom: 20px;'>🌍 Navigation</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio("Go to", ["Quiz", "Flashcards", "Progress"], label_visibility="collapsed")
    st.session_state.current_page = page

    # Language options
    global_languages = {
        "English": "English",
        "European Languages": [
            "Spanish",
            "French",
            "German",
            "Italian",
            "Portuguese",
            "Russian",
            "Dutch",
            "Polish",
            "Ukrainian",
            "Romanian",
            "Greek",
            "Czech",
            "Swedish",
            "Norwegian",
            "Danish",
            "Finnish",
            "Hungarian",
            "Bulgarian",
        ],
        "Asian Languages": [
            "Chinese",
            "Japanese",
            "Korean",
            "Arabic",
            "Hebrew",
            "Turkish",
            "Thai",
            "Vietnamese",
            "Indonesian",
            "Malay",
            "Filipino",
            "Persian",
            "Hindi",
            "Tamil",
            "Telugu",
            "Kannada",
            "Malayalam",
            "Bengali",
            "Marathi",
            "Gujarati",
            "Punjabi",
            "Urdu",
        ],
        "Other Languages": [
            "Swahili",
            "Zulu",
            "Afrikaans",
            "Catalan",
            "Croatian",
            "Serbian",
            "Slovak",
            "Slovenian",
            "Lithuanian",
            "Latvian",
            "Estonian",
            "Maltese",
            "Icelandic",
        ],
    }

    lang_choice = st.sidebar.selectbox(
        "🌐 Choose Language",
        options=[global_languages["English"]]
        + global_languages["European Languages"]
        + global_languages["Asian Languages"]
        + global_languages["Other Languages"],
        index=0,
    )

    question_count = st.sidebar.slider(
        "Number of Questions", min_value=5, max_value=20, value=20
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
    <div style='background: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #38ef7d; color: #ccc;'>
        <h4 style='color: #38ef7d;'>📚 How to use:</h4>
        <p>1. Upload a PDF document<br>
        2. Choose your language<br>
        3. Select number of questions<br>
        4. Start learning with quizzes or flashcards!</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])

    st.markdown(
        """
    <style>
    .stApp { background: #0f1116; color: #e0e0e0; }
    h1, h2, h3 { color: #38ef7d !important; font-weight: 600; }
    .stButton>button {
        background: rgba(56, 239, 125, 0.1);
        color: #38ef7d;
        border: 1px solid #38ef7d;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #38ef7d;
        color: #0f1116;
        border-color: #38ef7d;
        transform: translateY(-1px);
    }
    .flashcard-container {
        background: linear-gradient(135deg, rgba(56, 239, 125, 0.05) 0%, rgba(0, 0, 0, 0) 100%);
        border: 1px solid rgba(56, 239, 125, 0.3);
        border-radius: 16px;
        padding: 40px;
        margin: 20px auto;
        min-height: 250px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        transition: all 0.3s ease;
    }
    .progress-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .progress-card h2 {
        color: #38ef7d !important;
        font-size: 2em;
        margin: 10px 0;
    }
    .progress-card h3 {
        color: #ccc !important;
        font-size: 0.9em;
        margin: 5px 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    if uploaded_file:
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"

        if st.session_state.processed_file_id != current_file_id:
            st.session_state.processed_file_id = current_file_id
            st.cache_data.clear()

        with st.spinner("📄 Processing your document..."):
            raw_text, page_count = extract_text_from_pdf(uploaded_file)

        st.sidebar.markdown(
            f"""
        <div style='background: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #38ef7d; color: #ccc;'>
            <h4 style='color: #38ef7d; margin-bottom: 15px;'>📊 Document Info</h4>
            <p style='margin: 5px 0;'>📄 <strong>Pages:</strong> {page_count}</p>
            <p style='margin: 5px 0;'>📝 <strong>Characters:</strong> {len(raw_text)}</p>
            <p style='margin: 5px 0;'>🎯 <strong>Questions:</strong> {question_count}</p>
            <p style='margin: 5px 0;'>🌐 <strong>Language:</strong> {lang_choice}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if len(raw_text) > 50 and not raw_text.startswith("Error"):
            # Quiz Page
            if st.session_state.current_page == "Quiz":
                st.header("🎯 Quiz Mode")

                with st.spinner(
                    f"🔄 Generating {question_count} exam-style questions in {lang_choice}..."
                ):
                    mcqs = make_mcqs(raw_text, lang="English", max_questions=question_count)
                    mcqs = translate_full_json(mcqs, lang_choice)

                if mcqs:
                    st.success(f"✅ Generated {len(mcqs)} questions in {lang_choice}")

                    for i, q in enumerate(mcqs):
                        st.markdown(f"### ❓ Q{i+1}: {q['question']}")
                        if "difficulty" in q:
                            st.caption(
                                f"Difficulty: {q['difficulty'].capitalize()}"
                            )

                        key_answered = f"answered_{i}"
                        if key_answered not in st.session_state:
                            st.session_state[key_answered] = None

                        selected_option = st.radio(
                            f"Select an answer for Q{i+1}:",
                            q["options"],
                            key=f"radio_{i}",
                            index=None,
                            label_visibility="collapsed",
                        )

                        if selected_option:
                            if st.session_state[key_answered] is None:
                                # First time answering this question
                                is_correct = selected_option == q["answer"]
                                st.session_state[key_answered] = selected_option
                                if is_correct:
                                    st.success("🎉 Correct! Well done!")
                                else:
                                    st.error(
                                        f"❌ Incorrect! The correct answer is: **{q['answer']}**"
                                    )
                                update_progress(correct=is_correct)
                            else:
                                # Already answered; don't double count
                                if selected_option == q["answer"]:
                                    st.info("You already answered this correctly.")
                                else:
                                    st.info("You already answered this question.")

                        st.markdown("---")
                else:
                    st.warning(
                        "⚠️ No questions could be generated from this PDF. Try a document with more educational content."
                    )

            # Flashcards Page
            elif st.session_state.current_page == "Flashcards":
                st.header("📚 Flashcard Mode")

                flashcards = make_flashcards(
                    raw_text, lang=lang_choice, max_cards=question_count
                )

                if flashcards:
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col2:
                        current_card = flashcards[st.session_state.flashcard_index]
                        card_content = (
                            current_card["answer"]
                            if st.session_state.show_answer
                            else current_card["question"]
                        )

                        st.markdown(
                            f'<div class="flashcard-container">'
                            f'<div style="font-size: 1.3em; color: #e0e0e0; font-weight: 500;">{card_content}</div>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        if st.button(
                            "🔄 Flip Card",
                            key=f"flip_{st.session_state.flashcard_index}",
                            use_container_width=True,
                        ):
                            st.session_state.show_answer = not st.session_state.show_answer
                            update_progress(flashcard_studied=True)

                        nav_col1, nav_col2, nav_col3 = st.columns([1, 1, 1])

                        with nav_col1:
                            if st.button("⬅️ Previous", use_container_width=True):
                                st.session_state.flashcard_index = (
                                    st.session_state.flashcard_index - 1
                                ) % len(flashcards)
                                st.session_state.show_answer = False

                        with nav_col2:
                            st.markdown(
                                f"<p style='text-align: center; color: #ccc;'>📖 Card {st.session_state.flashcard_index + 1} of {len(flashcards)}</p>",
                                unsafe_allow_html=True,
                            )

                        with nav_col3:
                            if st.button("Next ➡️", use_container_width=True):
                                st.session_state.flashcard_index = (
                                    st.session_state.flashcard_index + 1
                                ) % len(flashcards)
                                st.session_state.show_answer = False

                    st.info("💡 Use the Flip Card button to reveal the answer!")

                else:
                    st.warning(
                        "⚠️ No flashcards could be generated from this PDF."
                    )

            # Progress Page
            elif st.session_state.current_page == "Progress":
                st.header("📊 Your Learning Progress")
                progress = st.session_state.quiz_progress

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(
                        f"""
                    <div class="progress-card">
                        <h3>Total Questions</h3>
                        <h2>{progress['total_questions']}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    accuracy = (
                        progress["correct_answers"]
                        / progress["total_questions"]
                        * 100
                        if progress["total_questions"] > 0
                        else 0
                    )
                    st.markdown(
                        f"""
                    <div class="progress-card">
                        <h3>Accuracy</h3>
                        <h2>{accuracy:.1f}%</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col3:
                    st.markdown(
                        f"""
                    <div class="progress-card">
                        <h3>Correct Answers</h3>
                        <h2>{progress['correct_answers']}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col4:
                    st.markdown(
                        f"""
                    <div class="progress-card">
                        <h3>Flashcards Studied</h3>
                        <h2>{progress['flashcards_studied']}</h2>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                if progress["total_questions"] > 0:
                    col1, col2 = st.columns(2)

                    with col1:
                        accuracy = (
                            progress["correct_answers"]
                            / progress["total_questions"]
                            * 100
                        )
                        fig_gauge = go.Figure(
                            go.Indicator(
                                mode="gauge+number+delta",
                                value=accuracy,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={
                                    "text": "Accuracy Score",
                                    "font": {"color": "#38ef7d", "size": 20},
                                },
                                delta={
                                    "reference": 50,
                                    "increasing": {"color": "#38ef7d"},
                                },
                                gauge={
                                    "axis": {
                                        "range": [None, 100],
                                        "tickwidth": 1,
                                        "tickcolor": "#38ef7d",
                                    },
                                    "bar": {"color": "#38ef7d"},
                                    "bgcolor": "rgba(255, 255, 255, 0.05)",
                                    "borderwidth": 2,
                                    "bordercolor": "#38ef7d",
                                    "steps": [
                                        {
                                            "range": [0, 50],
                                            "color": "rgba(255, 107, 107, 0.3)",
                                        },
                                        {
                                            "range": [50, 80],
                                            "color": "rgba(255, 193, 7, 0.3)",
                                        },
                                        {
                                            "range": [80, 100],
                                            "color": "rgba(56, 239, 125, 0.3)",
                                        },
                                    ],
                                    "threshold": {
                                        "line": {"color": "#38ef7d", "width": 4},
                                        "thickness": 0.75,
                                        "value": 90,
                                    },
                                },
                            )
                        )
                        fig_gauge.update_layout(
                            height=300,
                            paper_bgcolor="rgba(0,0,0,0)",
                            font={"color": "#e0e0e0"},
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col2:
                        if (
                            progress["correct_answers"] > 0
                            or progress["incorrect_answers"] > 0
                        ):
                            fig_pie = px.pie(
                                values=[
                                    progress["correct_answers"],
                                    progress["incorrect_answers"],
                                ],
                                names=["Correct ✅", "Incorrect ❌"],
                                title="Answer Distribution",
                                color=["Correct ✅", "Incorrect ❌"],
                                color_discrete_map={
                                    "Correct ✅": "#38ef7d",
                                    "Incorrect ❌": "#ff6b6b",
                                },
                            )
                            fig_pie.update_traces(textinfo="percent+label")
                            fig_pie.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                font={"color": "#e0e0e0"},
                                height=300,
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                if st.button("🔄 Reset Progress", type="secondary", use_container_width=True):
                    st.session_state.quiz_progress = {
                        "total_questions": 0,
                        "correct_answers": 0,
                        "incorrect_answers": 0,
                        "quizzes_taken": 0,
                        "flashcards_studied": 0,
                    }
                    st.rerun()

        else:
            st.error(
                "❌ Could not extract enough text from this PDF. Please try a different document."
            )

    else:
        st.info("👆 Please upload a PDF document to get started!")
        st.markdown(
            """
        <div style='background: rgba(56, 239, 125, 0.05); border: 1px solid rgba(56, 239, 125, 0.3); border-radius: 16px; padding: 40px; text-align: center; margin: 20px 0;'>
            <h2 style='color: #38ef7d; margin-bottom: 20px;'>🌿 Welcome to Quillium!</h2>
            <p style='color: #ccc; font-size: 1.1em; line-height: 1.6;'>
            Upload a PDF to create interactive quizzes and flashcards in your preferred language.
            <br>Supports <strong>50+ languages</strong> including Spanish, French, German, Chinese, Arabic, and many more!
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Persist full traceback to a file for debugging when Streamlit swallows it
        import traceback

        tb = traceback.format_exc()
        try:
            with open("app_error.log", "w", encoding="utf-8") as fh:
                fh.write(tb)
        except Exception:
            pass
        raise
