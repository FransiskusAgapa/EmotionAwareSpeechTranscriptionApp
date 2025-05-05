# ===============================
# Emotion-Aware Speech-to-Text App using Fine-Tuned Wav2Vec2
# ===============================

# --- Import Required Libraries ---
import streamlit as st  # Builds the simple web app interface
import torch            # Runs the AI models (transcription + emotion)
import torchaudio       # Helps load and process audio files

# Hugging Face tools to process and run models
from transformers import (
    Wav2Vec2Processor,                   # For converting sound into text-friendly format
    Wav2Vec2FeatureExtractor,            # For preparing audio for emotion model
    Wav2Vec2ForCTC,                      # Pretrained model for speech-to-text
    Wav2Vec2ForSequenceClassification    # Model used for classifying emotion from audio
)

# --- Setup Audio Backend ---
torchaudio.set_audio_backend("soundfile")  # Makes sure audio loads properly on all systems

# --- Load Speech-to-Text Model ---
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  # Turns sound into numbers for model
transcription_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")  # Model that turns speech into text

# --- Load Fine-Tuned Emotion Classifier ---
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./emotion_model")  # Prepares audio for emotion model
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("./emotion_model")  # Loads your fine-tuned emotion model

# --- Define Emotion Labels ---
label2id = {'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 'angry': 4, 'fearful': 5, 'disgust': 6, 'surprised': 7}  # Map labels to numbers
id2label = {v: k for k, v in label2id.items()}  # Map numbers back to emotion names

# --- Streamlit Web UI ---
st.title("Speak to Text + Emotion")  # Big title on the web app
st.write("Upload a WAV audio file. I’ll transcribe the text and tell you the emotion.")  # Instructions

# --- Upload Audio File ---
audio_file = st.file_uploader("Upload your voice recording (WAV format only)", type=["wav"])  # Upload box

# --- Transcribe Function (speech → text) ---
def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)  # Load the audio and its sample rate

    if sample_rate != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)  # Match expected sample rate
        waveform = resample(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)  # If stereo, keep just one channel

    waveform = waveform.squeeze().unsqueeze(0)  # Format to [1, length] for model

    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")  # Convert audio to model input
    input_values = inputs["input_values"]  # Get the input data for the model

    with torch.no_grad():  # No need to calculate gradients since we're not training
        logits = transcription_model(input_values).logits  # Get model predictions

    predicted_ids = torch.argmax(logits, dim=-1)  # Pick the most likely prediction for each part
    return processor.decode(predicted_ids[0])  # Turn predictions into text

# --- Emotion Detection Function (speech → emotion) ---
def predict_emotion(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)  # Load audio file

    if sample_rate != 16000:
        resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)  # Resample to 16kHz
        waveform = resample(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)  # Keep mono channel only

    inputs = feature_extractor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")  # Prepare audio

    with torch.no_grad():  # Inference mode
        logits = emotion_model(**inputs).logits  # Run emotion model

    predicted_id = torch.argmax(logits, dim=-1).item()  # Get the predicted label ID
    return id2label[predicted_id]  # Return emotion name

# --- When Audio is Uploaded ---
if audio_file:
    temp_path = "temp_audio.wav"  # Temporary filename
    with open(temp_path, "wb") as f:
        f.write(audio_file.read())  # Save uploaded file locally

    result_text = transcribe(temp_path)       # Transcribe audio to text
    result_emotion = predict_emotion(temp_path)  # Predict emotion

    # Show results
    st.subheader("Transcribed Text:")
    st.success(result_text)  # Display the text

    st.subheader("Detected Emotion:")
    st.info(result_emotion)  # Display the emotion
