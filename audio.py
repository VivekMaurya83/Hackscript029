from flask import Flask, request, jsonify, render_template
import soundfile as sf
import speech_recognition as sr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
from pydub.silence import detect_silence
import Levenshtein
from textblob import TextBlob
from better_profanity import profanity
import librosa
import numpy as np
import os

app = Flask(__name__)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Aggressive keywords
AGGRESSIVE_KEYWORDS = ["angry", "frustrated", "stop", "why", "problem", "hate", "never", "worst", "bad", "ridiculous"]

# Load dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to convert audio to text
def audio_to_text(audio_file):
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API request failed"

# Function to differentiate user query and chatbot response
def differentiate_query_response(audio_file):
    # Load audio file
    audio = AudioSegment.from_wav(audio_file)
    
    # Split audio into chunks based on silence
    chunks = detect_silence(audio, min_silence_len=500, silence_thresh=-40)
    
    # Separate user query and chatbot response
    user_query = audio[:chunks[0][1]]  # First chunk is user query
    chatbot_response = audio[chunks[0][1]:]  # Rest is chatbot response
    
    # Save separated audio files
    user_query.export("user_query.wav", format="wav")
    chatbot_response.export("chatbot_response.wav", format="wav")
    
    return "user_query.wav", "chatbot_response.wav"

# Function to find the most similar question in the dataset
def find_most_similar_question(query, dataset):
    vectorizer = TfidfVectorizer()
    all_texts = list(dataset['Question']) + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    most_similar_index = cosine_similarities.argmax()
    most_similar_question = dataset.iloc[most_similar_index]['Question']
    most_similar_answer = dataset.iloc[most_similar_index]['Answer']
    similarity_score = cosine_similarities[0][most_similar_index]
    return most_similar_question, most_similar_answer, similarity_score

# Function to calculate accuracy
def calculate_accuracy(transcribed_text, correct_text):
    distance = Levenshtein.distance(transcribed_text.lower(), correct_text.lower())
    max_length = max(len(transcribed_text), len(correct_text))
    accuracy = (1 - distance / max_length) * 100
    return accuracy

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# Function to detect bad words
def detect_bad_words(text):
    profanity.load_censor_words()
    has_bad_words = profanity.contains_profanity(text)
    if has_bad_words:
        censored_text = profanity.censor(text)
        return True, censored_text
    return False, None

# Function to count long pauses in audio
def count_long_pauses(audio_file, silence_threshold=-40, min_silence_len=1000):
    audio = AudioSegment.from_wav(audio_file)
    silence_ranges = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_threshold)
    long_pauses = [(start, end) for (start, end) in silence_ranges if (end - start) >= 2000]  # 2 seconds
    return len(long_pauses)

# Function to detect aggressive tone in text
def detect_aggressive_tone_text(text):
    # Check for aggressive keywords
    aggressive_words = [word for word in AGGRESSIVE_KEYWORDS if word in text.lower()]
    if aggressive_words:
        return True, f"Aggressive keywords detected: {', '.join(aggressive_words)}"
    
    # Check for highly negative sentiment
    polarity, _ = analyze_sentiment(text)
    if polarity < -0.6:  # Highly negative sentiment
        return True, "Highly negative sentiment detected."
    
    return False, None

# Function to detect aggressive tone in audio
def detect_aggressive_tone_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract features
    pitch = librosa.yin(y, fmin=80, fmax=400)  # Detect pitch
    rms_energy = librosa.feature.rms(y=y)  # Detect volume (energy)
    speech_rate = len(y) / len(librosa.effects.split(y, top_db=30))  # Approximate speech rate
    
    # Check for aggressive tone
    if np.mean(pitch) > 200:  # High pitch
        return True, "High pitch detected (possible aggressive tone)."
    if np.mean(rms_energy) > 0.1:  # Loud volume
        return True, "Loud volume detected (possible aggressive tone)."
    if speech_rate > 0.5:  # Fast speech rate
        return True, "Fast speech rate detected (possible aggressive tone)."
    
    return False, None

# Flask route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle audio upload and analysis
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'user_audio' not in request.files:
        return jsonify({"error": "User audio file is required"}), 400
    
    user_file = request.files['user_audio']
    
    if user_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save the uploaded file
    user_audio_path = "user_audio.wav"
    user_file.save(user_audio_path)
    
    # Differentiate user query and chatbot response
    user_query_path, chatbot_response_path = differentiate_query_response(user_audio_path)
    
    # Convert audio to text
    user_query_text = audio_to_text(user_query_path)
    chatbot_response_text = audio_to_text(chatbot_response_path)
    
    # Load dataset
    dataset = load_dataset('asus_faq_with_categories.csv')
    
    # Find the most similar question and answer in the dataset
    most_similar_question, most_similar_answer, similarity_score = find_most_similar_question(user_query_text, dataset)
    
    # Perform analysis
    results = {
        "user_query": user_query_text,
        "chatbot_response": chatbot_response_text,
        "most_similar_question": most_similar_question,
        "most_similar_answer": most_similar_answer,
        "accuracy": calculate_accuracy(chatbot_response_text, most_similar_answer),
        "long_pauses": count_long_pauses(chatbot_response_path),
        "bad_words": detect_bad_words(chatbot_response_text),
        "sentiment": analyze_sentiment(chatbot_response_text),
        "aggressive_tone_text_user": detect_aggressive_tone_text(user_query_text),
        "aggressive_tone_text_chatbot": detect_aggressive_tone_text(chatbot_response_text),
        "aggressive_tone_audio_user": detect_aggressive_tone_audio(user_query_path),
        "aggressive_tone_audio_chatbot": detect_aggressive_tone_audio(chatbot_response_path),
    }
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True,port=5002)