# @title Default title text
# ==========================================================
# Advanced NLP Chatbot with Context Memory + Voice + Wikipedia
# ==========================================================

import nltk
import wikipedia
import speech_recognition as sr
import pyttsx3
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- NLTK Downloads ---------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# -------------------- Voice Engine -----------------------
engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------------------- Wikipedia --------------------------
wikipedia.set_lang("en")

# -------------------- Context Memory ---------------------
context_memory = {
    "last_topic": None
}

# -------------------- Knowledge Base ---------------------
knowledge_base = {
    "what is python": "Python is a high-level programming language used in web development, AI, data science, and automation.",
    "what is nlp": "Natural Language Processing enables machines to understand and respond to human language.",
    "what is machine learning": "Machine learning allows systems to learn from data and improve automatically."
}

# -------------------- NLP Preprocessing ------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t.isalnum() and t not in stop_words
    ]
    return " ".join(tokens)

# -------------------- Clean Question for Wikipedia -------
def extract_topic(question):
    question = question.lower()
    question = re.sub(r"(what is|who is|tell me about|explain|define|describe)", "", question)
    return question.strip()

# -------------------- Vectorize Knowledge Base -----------
kb_questions = list(knowledge_base.keys())
kb_vectors = TfidfVectorizer().fit_transform(
    [preprocess(q) for q in kb_questions]
)

vectorizer = TfidfVectorizer()
vectorizer.fit([preprocess(q) for q in kb_questions])

# -------------------- Wikipedia Search -------------------
def wikipedia_response(query):
    try:
        clean_query = extract_topic(query)
        context_memory["last_topic"] = clean_query
        return wikipedia.summary(clean_query, sentences=2)

    except wikipedia.DisambiguationError as e:
        return f"Be more specific. Try: {e.options[:2]}"

    except wikipedia.PageError:
        return "I couldn't find information on that topic."

    except:
        return "Error while fetching information."

# -------------------- Speech Recognition -----------------
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You:", text)
        return text
    except:
        return ""

# -------------------- Chatbot Brain ----------------------
def chatbot_response(user_input):
    processed = preprocess(user_input)

    # Follow-up handling
    if user_input.lower().startswith(("tell me more", "continue", "explain more")):
        if context_memory["last_topic"]:
            return wikipedia_response(context_memory["last_topic"])

    # Knowledge base similarity
    user_vector = vectorizer.transform([processed])
    similarity = cosine_similarity(user_vector, kb_vectors)
    best_match = similarity.argmax()

    if similarity[0][best_match] > 0.25:
        context_memory["last_topic"] = kb_questions[best_match]
        return knowledge_base[kb_questions[best_match]]

    # Open-domain Wikipedia (always try)
    return wikipedia_response(user_input)

# -------------------- Main Program -----------------------
print("\n🤖 Advanced Chatbot (Open Questions + Context + Voice)")
print("Ask anything | Say 'exit' to stop\n")

speak("Hello! I am your AI assistant. How can I help you?")

while True:
    mode = input("Press T for Text | V for Voice | E to Exit: ").lower()

    if mode == "e":
        speak("Goodbye!")
        print("Bot: Goodbye!")
        break

    if mode == "v":
        user_input = listen()
    else:
        user_input = input("You: ")

    if user_input.lower() == "exit":
        speak("Goodbye!")
        print("Bot: Goodbye!")
        break

    response = chatbot_response(user_input)
    print("Bot:", response)
    speak(response)
