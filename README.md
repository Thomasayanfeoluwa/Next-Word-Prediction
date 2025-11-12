# 🔮 Next‑Word Prediction  
*An NLP‑powered tool to predict the most likely next word(s) given a text sequence.*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/idowu-thomas-56819433b)  
[![Email](https://img.shields.io/badge/Email-ayanfeoluwadegoke@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:ayanfeoluwadegoke@gmail.com)  
[![Built With Python](https://img.shields.io/badge/Built%20With-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)  
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-4CAF50?style=for-the-badge&logo=streamlit&logoColor=white)](https://next-word-prediction-h8fmojheupyxavdrrpc4ma.streamlit.app/)

---

## ✨ Project Overview  
This project leverages advanced NLP techniques (RNNs/LSTMs/Transformers) to predict the next word in a sequence — useful for autocomplete, chatbots, and language tools.

## 🛠 Built With  
| Layer                | Technology                                           |
|------------------------|------------------------------------------------------|
| Text Preparation       | Tokenizer, Padding, Sequence Generation               |
| Modeling               | RNN, LSTM, Bi‑LSTM, Transformer                       |
| UI / Deployment        | Streamlit, Docker, AWS Lambda (optional)             |
| Tracking               | MLflow                                                |

## 🔍 Key Features  
- Interactive next‑word suggestion UI  
- Customizable input length & model settings  
- Visual confidence ranking and word‑cloud of suggestions  

## 🚀 Get Started Locally  
```bash
git clone https://github.com/Thomasayanfeoluwa/Next-Word-Prediction
cd next‑word‑prediction  
python3 -m venv venv  
source venv/bin/activate  
pip install -r requirements.txt  
streamlit run app.py
