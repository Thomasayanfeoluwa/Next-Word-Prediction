import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Paths to your model and tokenizer
# -------------------------------
model_path = "best_next_word_lstm.h5"
tokenizer_path = "tokenizer.pkl"

# Load model and tokenizer
try:
    model = load_model(model_path)
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("Model or tokenizer not found. Please ensure the paths are correct.")
    st.stop()

max_sequence_len = model.input_shape[1] + 1  # matches training sequence length

# -------------------------------
# Enhanced Prediction Function
# -------------------------------
def predict_next_words(model, tokenizer, text, max_sequence_len, num_predictions=5, temperature=1.0, top_k=50):
    """
    Predicts the next words with temperature sampling and top-k filtering.
    """
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    
    # Trim or pad token_list
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Predict probabilities
    predicted_probs = model.predict(token_list, verbose=0)[0]
    
    # Apply top-k filtering
    top_k_indices = np.argsort(predicted_probs)[-top_k:]
    top_k_probs = predicted_probs[top_k_indices]
    
    # Apply temperature sampling
    top_k_probs = np.log(top_k_probs) / temperature
    top_k_probs = np.exp(top_k_probs) / np.sum(np.exp(top_k_probs))
    
    # Sample next word indices
    next_word_indices = np.random.choice(top_k_indices, size=num_predictions, p=top_k_probs)
    
    # Map indices to words
    next_words = []
    for index in next_word_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                next_words.append(word)
                break
    return next_words

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("Next-Word Prediction with Temperature Sampling")

st.sidebar.header("Prediction Parameters")
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
top_k = st.sidebar.slider("Top-K", 10, 100, 50, 10)
num_predictions = st.sidebar.slider("Number of Predictions", 1, 10, 5, 1)

input_text = st.text_input("Enter a phrase:", "The quick brown fox")

if st.button("Predict Next Word"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting..."):
            next_words = predict_next_words(model, tokenizer, input_text, max_sequence_len, num_predictions, temperature, top_k)
            st.success("Predicted next words:")
            st.write(next_words)
