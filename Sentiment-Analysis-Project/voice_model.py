import streamlit as st
import whisper
from transformers import pipeline
import tempfile

# Load models
st.title("🎙️ Speech Sentiment Analysis")
st.write("Analyze sentiment from audio recordings.")

model = whisper.load_model("base")
sentiment_analysis = pipeline(
    "sentiment-analysis",
    framework="pt",
    model="SamLowe/roberta-base-go_emotions"
)

# Helper functions
def analyze_sentiment(text):
    results = sentiment_analysis(text)
    sentiment_results = {
        result['label']: result['score'] for result in results
    }
    return sentiment_results

def get_sentiment_emoji(sentiment):
    emoji_mapping = {
        "disappointment": "😞",
        "sadness": "😢",
        "annoyance": "😠",
        "neutral": "😐",
        "disapproval": "👎",
        "realization": "😮",
        "nervousness": "😬",
        "approval": "👍",
        "joy": "😄",
        "anger": "😡",
        "embarrassment": "😳",
        "caring": "🤗",
        "remorse": "😔",
        "disgust": "🤢",
        "grief": "😥",
        "confusion": "😕",
        "relief": "😌",
        "desire": "😍",
        "admiration": "😌",
        "optimism": "😊",
        "fear": "😨",
        "love": "❤️",
        "excitement": "🎉",
        "curiosity": "🤔",
        "amusement": "😄",
        "surprise": "😲",
        "gratitude": "🙏",
        "pride": "🦁"
    }
    return emoji_mapping.get(sentiment, "")

def display_sentiment_results(sentiment_results, option):
    sentiment_text = ""
    for sentiment, score in sentiment_results.items():
        emoji = get_sentiment_emoji(sentiment)
        if option == "Sentiment Only":
            sentiment_text += f"{sentiment} {emoji}\n"
        elif option == "Sentiment + Score":
            sentiment_text += f"{sentiment} {emoji}: {score:.2f}\n"
    return sentiment_text

def inference(audio_file, sentiment_option):
    # Save the UploadedFile object to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(audio_file.read())
        temp_path = temp_file.name

    audio = whisper.load_audio(temp_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    lang = max(probs, key=probs.get)

    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(model, mel, options)

    sentiment_results = analyze_sentiment(result.text)
    sentiment_output = display_sentiment_results(sentiment_results, sentiment_option)

    return lang.upper(), result.text, sentiment_output

# Streamlit UI
st.sidebar.title("Options")
sentiment_option = st.sidebar.radio(
    "Display Options",
    ("Sentiment Only", "Sentiment + Score")
)

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with st.spinner("Processing..."):
        try:
            lang, transcript, sentiment_output = inference(uploaded_file, sentiment_option)
            st.subheader("Results")
            st.write(f"**Detected Language:** {lang}")
            st.write(f"**Transcript:** {transcript}")
            st.write("**Sentiments:**")
            st.text(sentiment_output)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload an audio file to start sentiment analysis.")
