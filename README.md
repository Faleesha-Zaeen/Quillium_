# 🪶 Quillium

**Quillium** is your AI-powered intelligent question and flashcard generator.  
Just upload your PDFs or paste text, and it will transform your notes into interactive quizzes and flashcards.

---

##  Current Features

-  Upload PDFs to generate quizzes and flashcards  
-  AI-powered MCQ generation using OpenRouter (DeepSeek model)  
-  Smart distractors for MCQs to make quizzes challenging  
-  Flashcard mode with click-to-flip functionality  
-  Optional AI translation into 50+ languages (via local model + OpenRouter)  
-  Track your learning progress with interactive stats and charts  
-  Minimalist, elegant green-themed UI for a pleasant learning experience  

---

##  Features in Detail

###  MCQs
-  Auto-generated from PDFs using an LLM (via OpenRouter)  
-  Includes smart distractors and difficulty labels  
-  Click an answer to reveal correctness (with instant feedback)  

###  Flashcards
-  Flip to reveal answers  
-  Navigate between cards with next/previous buttons  
 -  Generated from the same MCQ engine to ensure coverage of key concepts  

###  Progress Dashboard
-  Tracks total questions attempted, accuracy, correct answers, and flashcards studied  
-  Visualizes performance with interactive charts  

---

##  Notes
-  PDFs should be text-based, not scanned images  
-  Large PDFs may take longer to process  
-  First-time runs download AI translation models (~hundreds of MB)  
-  MCQ and JSON-level translations require an OpenRouter API key  

Supported languages include (but are not limited to): English, Spanish, French, German, Italian, Portuguese, Russian, Dutch, Polish, Ukrainian, Romanian, Greek, Czech, Swedish, Norwegian, Danish, Finnish, Hungarian, Bulgarian, Chinese, Japanese, Korean, Arabic, Turkish, Thai, Vietnamese, Indonesian, Malay, Filipino, Persian, Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, Urdu, and other European/Asian languages (50+ in total).  

---
##  Configuration

- **Python environment**: Install dependencies with:

	```powershell
	pip install -r requirements.txt
	```

- **OpenRouter API**: Set your API key before running the app:

	```powershell
	$env:OPENROUTER_API_KEY = "YOUR_KEY_HERE"
	```

	Quillium uses `deepseek/deepseek-r1:7b` via OpenRouter for MCQ generation and JSON-level translation.

- **Run the app** (from the project root):

	```powershell
	streamlit run app.py
	```
