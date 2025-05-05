
# Emotion-Aware Speech Transcription App

This is a simple app that can **listen to your voice**, **turn it into text**, and also **guess how you're feeling** based on how you sound.

It uses powerful AI models to:

1. **Transcribe speech to text** (what you say)
2. **Detect emotion** in your voice (how you say it)

---

## What This App Can Do

* You record or upload a `.wav` audio file of your voice.
* The app writes out what you said.
* It also tells you what emotion it thinks you're expressing, like happy, sad, angry, or calm.

---

## Why This Is Cool

Big tech companies like Facebook, Apple, Amazon, Netflix, and Google all use AI that understands not just words, but emotions too. This project shows how you can build your own version.

---

## How It Works

* It uses **Wav2Vec2**, a machine learning model from Facebook that understands speech.
* The model is trained to turn sound into words (transcription).
* A second model is trained to recognize emotion from the tone of your voice.
* Both models work together to give the final result.

---

## What You Need to Run It

* Python 3.8 or higher
* A modern browser
* Basic understanding of files and folders
* `.wav` audio files (recordings of your voice)

---

## Tech Stack

* **Streamlit** for the web interface
* **Wav2Vec2** for speech-to-text
* **Wav2Vec2ForSequenceClassification** for emotion detection
* **Hugging Face Transformers** and **Datasets** libraries
* **PyTorch** for deep learning
* **Torchaudio** for working with audio files

---

## Future Ideas

* Let the app talk back with different voices depending on your emotion.
* Connect it to smart devices (turn off lights if you're tired).
* Train it on your own voice to personalize the results.

---
