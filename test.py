import streamlit as st
import sounddevice as sd
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
import time
from autocorrect import Speller
import mysql.connector
from datetime import datetime
import matplotlib.pyplot as plt  # For plotting the accuracy graph
from pydub import AudioSegment
import wave



# MySQL Database Configuration
db_config = {
    "host": "localhost",  # Replace with your MySQL host
    "user": "root",       # Replace with your MySQL username
    "password": "@Vivek1108",  # Replace with your MySQL password
    "database": "qa1"      # Replace with your database name
}

# Function to connect to MySQL
def connect_to_mysql():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        st.error(f"Error connecting to MySQL: {err}")
        return None

# Function to fetch conversation metrics from MySQL
def fetch_conversation_metrics():
    conn = connect_to_mysql()
    if conn:
        cursor = conn.cursor()
        query = "SELECT user_id, accuracy, timestamp FROM conversation_metrics"
        cursor.execute(query)
        data = cursor.fetchall()
        conn.close()
        return data
    return []

# Function to insert conversation metrics into MySQL
def insert_conversation_metrics(user_id, accuracy, timestamp):
    conn = connect_to_mysql()
    if conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO conversation_metrics (user_id, accuracy, timestamp)
        VALUES (%s, %s, %s)
        """
        cursor.execute(query, (user_id, accuracy, timestamp))
        conn.commit()
        conn.close()

# Initialize the recognizer
recognizer = sr.Recognizer()

# Initialize spell checker
spell = Speller()

# Define domain-related keywords
DOMAIN_KEYWORDS = ["asus", "laptop", "computer", "hardware", "software", "battery", "screen", "keyboard", "warranty", "support"]

# Define aggressive tone keywords
AGGRESSIVE_KEYWORDS = ["angry", "frustrated", "hate", "stupid", "idiot", "annoying", "useless", "worst", "terrible", "disgusting"]

# Function to capture audio
def capture_audio(filename, duration=5, samplerate=16000):
    st.write(f"Recording {filename}...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, audio, samplerate)
    st.write("Recording stopped.")

# Function to convert audio to text
def audio_to_text(audio_file):
    with sr.AudioFile(audio_file) as source:
        # Adjust for ambient noise and set energy threshold
        recognizer.adjust_for_ambient_noise(source)
        recognizer.energy_threshold = 4000  # Adjust based on environment
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        # Correct spelling errors in the transcribed text
        corrected_text = spell(text)
        return corrected_text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API request failed"

# Function to load dataset
def load_dataset(file_path):
    try:
        dataset = pd.read_csv(file_path)
        if dataset.empty:
            st.warning("Warning: The dataset is empty.")
        return dataset
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()  # Return an empty DataFrame

# Function to find the most similar question
def find_most_similar_question(query, dataset):
    if dataset.empty:
        return "No dataset loaded", "No dataset loaded"
    
    vectorizer = TfidfVectorizer()
    all_texts = list(dataset['Question']) + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    if cosine_similarities.size == 0:
        return "No similar question found", "No similar answer found"
    
    most_similar_index = cosine_similarities.argmax()
    most_similar_score = cosine_similarities.max()  # Get the similarity score
    
    # If the similarity score is too low, consider it a no-match
    if most_similar_score < 0.2:  # Threshold for considering a match
        return "No similar question found", "No similar answer found"
    
    most_similar_question = dataset.iloc[most_similar_index]['Question']
    most_similar_answer = dataset.iloc[most_similar_index]['Answer']
    
    return most_similar_question, most_similar_answer

# Function to calculate accuracy
def calculate_accuracy(transcribed_text, correct_text):
    if correct_text == "No similar answer found":
        return 0  # 0% accuracy if no match is found
    
    distance = Levenshtein.distance(transcribed_text.lower(), correct_text.lower())
    max_length = max(len(transcribed_text), len(correct_text))
    if max_length == 0:
        return 0  # Avoid division by zero
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

# Function to check if the text is relevant to the domain
def is_relevant_to_domain(text, domain_keywords):
    text_lower = text.lower()
    for keyword in domain_keywords:
        if keyword in text_lower:
            return True
    return False

# Function to detect aggressive tone
def detect_aggressive_tone(text):
    # Check for aggressive keywords
    text_lower = text.lower()
    has_aggressive_keywords = any(keyword in text_lower for keyword in AGGRESSIVE_KEYWORDS)
    
    # Analyze sentiment
    polarity, _ = analyze_sentiment(text)
    
    # If both conditions are met, flag as aggressive
    if has_aggressive_keywords and polarity < -0.5:  # Highly negative sentiment
        return True
    return False

# Initialize session state
if 'user_query_text' not in st.session_state:
    st.session_state.user_query_text = None
if 'system_answer_text' not in st.session_state:
    st.session_state.system_answer_text = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []  # To store conversation history
if 'accuracy_history' not in st.session_state:
    st.session_state.accuracy_history = []  # To store accuracy history

# Streamlit App
def main():
    st.title("ASUS FAQ Chatbot")

    # Sidebar for Admin Panel
    st.sidebar.title("User Panel")

    # Load dataset
    dataset = load_dataset('asus_faq_with_categories.csv')

    # Chatbot Interface
    st.header("Chatbot Interface")

    # Record user query
    if st.button("Record User Query"):
        user_audio_file = "user_audio.wav"
        capture_audio(user_audio_file, duration=12)  # Capture 12 seconds of user audio

        # Convert user audio to text
        st.session_state.user_query_text = audio_to_text(user_audio_file)
        st.write(f"User Query: {st.session_state.user_query_text}")

        # Check if user query is relevant to the domain
        if not is_relevant_to_domain(st.session_state.user_query_text, DOMAIN_KEYWORDS):
            st.warning("Alert: User query is not relevant to the domain!")

        # Check for aggressive tone in user query
        if detect_aggressive_tone(st.session_state.user_query_text):
            st.warning("Alert: Aggressive tone detected in user query!")

    # Record system response
    if st.button("Record System Response"):
        if st.session_state.user_query_text is None:
            st.warning("Please record a user query first.")
        else:
            system_audio_file = "system_audio.wav"
            capture_audio(system_audio_file, duration=12)  # Capture 12 seconds of system audio

            # Convert system audio to text
            st.session_state.system_answer_text = audio_to_text(system_audio_file)
            st.write(f"System Answer: {st.session_state.system_answer_text}")

            # Check if system answer is relevant to the domain
            if not is_relevant_to_domain(st.session_state.system_answer_text, DOMAIN_KEYWORDS):
                st.warning("Alert: System answer is not relevant to the domain!")

            # Check for aggressive tone in system answer
            if detect_aggressive_tone(st.session_state.system_answer_text):
                st.warning("Alert: Aggressive tone detected in system answer!")

            # Find the most similar question and answer
            most_similar_question, most_similar_answer = find_most_similar_question(st.session_state.user_query_text, dataset)
            st.write(f"Most Similar Question in Dataset: {most_similar_question}")
            st.write(f"Answer in Dataset: {most_similar_answer}")

            # Calculate accuracy
            accuracy = calculate_accuracy(st.session_state.system_answer_text, most_similar_answer)
            st.write(f"Answer Accuracy: {accuracy:.2f}%")

            # Perform sentiment analysis
            polarity, subjectivity = analyze_sentiment(st.session_state.system_answer_text)
            st.write(f"Sentiment Analysis - Polarity: {polarity}, Subjectivity: {subjectivity}")

            # Detect bad words
            has_bad_words, censored_text = detect_bad_words(st.session_state.system_answer_text)
            if has_bad_words:
                st.warning(f"Alert: Bad words detected in the system's answer! Censored Text: {censored_text}")
            else:
                st.success("No bad words detected in the system's answer.")

            # Insert conversation metrics into MySQL
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            insert_conversation_metrics("user123", accuracy, timestamp)
            st.success("Conversation metrics saved to the database!")

            # Update conversation history
            st.session_state.conversation_history.append({
                "User Query": st.session_state.user_query_text,
                "System Answer": st.session_state.system_answer_text,
                "Accuracy": accuracy,
                "Timestamp": timestamp
            })

            # Update accuracy history
            st.session_state.accuracy_history.append(accuracy)

    # Display Conversation History
    st.header("Conversation History")
    if st.session_state.conversation_history:
        history_df = pd.DataFrame(st.session_state.conversation_history)
        st.dataframe(history_df)
    else:
        st.write("No conversation history yet.")

    # Plot Accuracy Graph
    st.header("Accuracy Over Time")
    if st.session_state.accuracy_history:
        plt.figure(figsize=(10, 5))
        plt.plot(st.session_state.accuracy_history, marker='o')
        plt.title("Accuracy Over Time")
        plt.xlabel("Conversation Number")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        st.pyplot(plt)
    else:
        st.write("No accuracy data to display yet.")

# Run the Streamlit app
if __name__ == "__main__":
    main()