import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Paths to model and tokenizer
# -------------------------------
MODEL_PATH = "best_next_word_lstm.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# -------------------------------
# Load Model and Tokenizer
# -------------------------------
try:
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("Error: Model or tokenizer not found. Please ensure the paths are correct.")
    st.stop()

# Reverse mapping from index to word for fast lookup
index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

# Maximum sequence length used during training
max_sequence_len = model.input_shape[1]

# -------------------------------
# Next-Word Prediction Function
# -------------------------------
def predict_next_words(
    model, tokenizer, text, max_sequence_len, num_predictions=3, temperature=0.8, top_k=30
):
    """
    Predicts the next words for a given input sequence using top-k filtering and temperature sampling.
    Returns a list of words sorted by predicted likelihood.
    """
    # Tokenize input text
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]

    # Trim or pad the token list to match model input
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    # Predict probability distribution for next word
    predicted_probs = model.predict(token_list, verbose=0)[0]

    # Top-k filtering
    top_k_indices = np.argsort(predicted_probs)[-top_k:]
    top_k_probs = predicted_probs[top_k_indices]

    # Temperature scaling
    scaled_probs = np.log(top_k_probs + 1e-8) / temperature
    scaled_probs = np.exp(scaled_probs)
    scaled_probs /= np.sum(scaled_probs)

    # Sample next words
    next_word_indices = np.random.choice(top_k_indices, size=num_predictions, p=scaled_probs)

    # Map indices to words
    next_words = [index_to_word.get(idx, "") for idx in next_word_indices]
    return next_words

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("Next Word Prediction Engine")
st.markdown(
    "Professional text prediction powered by an LSTM model. "
    "Adjust the settings in the sidebar to customize the output."
)

# Sidebar settings
st.sidebar.header("Prediction Settings")
temperature = st.sidebar.slider("Sampling Temperature", 0.1, 2.0, 0.8, 0.1)
top_k = st.sidebar.slider("Top-K Words Considered", 10, 100, 30, 5)
num_predictions = st.sidebar.slider("Number of Words to Predict", 1, 5, 3, 1)

# Input
input_text = st.text_input("Enter a text sequence for prediction:", "Thomas is going to the market")

# Prediction
if st.button("Predict"):
    if not input_text.strip():
        st.warning("Please enter some text to predict next words.")
    else:
        with st.spinner("Generating predictions..."):
            next_words = predict_next_words(
                model,
                tokenizer,
                input_text,
                max_sequence_len,
                num_predictions=num_predictions,
                temperature=temperature,
                top_k=top_k
            )
            st.success("Recommended Next Words:")
            # Display as bullet points for professional readability
            for word in next_words:
                st.write(f"• {word}")
