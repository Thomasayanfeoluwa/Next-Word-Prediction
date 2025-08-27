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
model = load_model(model_path)

with open(tokenizer_path, "rb") as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = model.input_shape[1] + 1  # matches training sequence length

# -------------------------------
# Prediction function
# -------------------------------
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text.lower()])[0]
    
    # Trim or pad token_list
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict
    predicted_probs = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probs, axis=1)[0]
    
    # Map index to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return "<OOV>"

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Next Word Prediction App")

input_text = st.text_input("Enter a phrase:")

if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        st.success(f"Predicted next word: {next_word}")
