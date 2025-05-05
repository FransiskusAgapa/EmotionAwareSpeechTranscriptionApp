# Simple Speech-to-Text App using Wav2Vec2 

# Import the tools we need
import streamlit as st              # Web app library for creating UI
import torch                        # Deep learning library - run the AI model that understands audio and makes predictions.
import torchaudio                   # For working with audio files - helps us load .wav files and modify them if needed (like changing sample rate).
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # Pretrained speech tools -  the actual tools from Hugging Face that turn speech into text. 

# Tell torchaudio to use the "soundfile" engine to load .wav files
torchaudio.set_audio_backend("soundfile")

# Load the processor to turn sound into numbers the model understands
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# Load the actual Wav2Vec2 model that can predict words from audio
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Set up the title and instructions for the web app
st.title("Speak to Text")  # Big title at the top of the app
st.write("Upload an audio file and I'll transcribe it to text.")  # Description

# File uploader lets user pick a .wav file to upload
audio_file = st.file_uploader("Upload your voice recording (WAV format only)", type=["wav"])

# This function takes in the audio file and returns the text it hears
def transcribe(audio_path):
    # Load the audio file into a waveform (sound signal) and its sample rate
    waveform, sample_rate = torchaudio.load(audio_path)

    # If audio is not 16k samples per second, convert it
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # If stereo (2 channels), keep only 1 channel
    if waveform.shape[0] > 1:
        waveform = waveform[0].unsqueeze(0)  # Make it mono

    # Ensure the shape is [1, length] (batch of 1 audio clip)
    waveform = waveform.squeeze().unsqueeze(0)

    # 🧪 Turn audio into model-ready input values
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    input_values = inputs['input_values']  # Extract just the audio values

    # 🚫 Don't calculate gradients (we're not training)
    with torch.no_grad():
        logits = model(input_values).logits  # Predict the raw scores

    # 🧠 Find the most likely ID (word piece) at each time step
    predicted_ids = torch.argmax(logits, dim=-1)

    # 🔤 Turn the IDs into readable text
    transcription = processor.decode(predicted_ids[0])
    return transcription  # Return the final text result

# ▶️ When the user uploads an audio file, run the transcription
if audio_file:
    # 💾 Save the uploaded audio file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(audio_file.read())

    # ✨ Run the transcription function
    result = transcribe(temp_path)

    # 📋 Show the transcribed text to the user
    st.subheader("📝 Transcribed Text:")
    st.success(result)

    # 🤖 Add a fun assistant reply based on the text
    st.subheader("🤖 Assistant says:")
    if "hello" in result.lower():
        st.info("Hey there! Great to hear from you 👋")
    elif "weather" in result.lower():
        st.info("Hmm, let me check the weather for you ☀️🌧️")
    else:
        st.info("Got it! Want to try another phrase?")

