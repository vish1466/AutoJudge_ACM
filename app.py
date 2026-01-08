import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack

# open all the models 
tfidf = joblib.load("models/tfidf.pkl")
clf = joblib.load("models/classifier.pkl")
reg = joblib.load("models/regressor.pkl")

# preprocess the text before hand
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# collect some additional features such as total_length and based on symbols as well
def extract_extra_features(full_text, desc, inp, out):
    
    total_length = len(full_text)
    desc_length = len(desc)
    input_length = len(inp)
    output_length = len(out)

    
    symbols = ['+', '-', '*', '/', '%', '=', '<', '>']
    num_digits = sum(c.isdigit() for c in full_text)
    num_symbols = sum(full_text.count(s) for s in symbols)

    
    return np.array([
        total_length,
        desc_length,
        input_length,
        output_length,
        num_digits,
        num_symbols
    ]).reshape(1, -1)

st.set_page_config(page_title="AutoJudge", layout="centered")
st.title(" AutoJudge: Programming Problem Difficulty Predictor")

st.write(
    "Paste a programming problem below to predict its **difficulty class** "
    "and **difficulty score**."
)

desc = st.text_area(" Problem Description")
inp = st.text_area(" Input Description")
out = st.text_area(" Output Description")

if st.button(" Predict Difficulty"):
    if not desc.strip():
        st.warning("Please enter the problem description.")
    else:
        
        # combine all the text into one part
        full_text = clean_text(desc + " " + inp + " " + out)

        # convert the text into embeddings based on the trained model
        X_tfidf = tfidf.transform([full_text])
        
        # extract the additional features as well 
        X_extra = extract_extra_features(
            full_text=full_text,
            desc=desc,
            inp=inp,
            out=out
        )

        # and add to the original set of features
        X_final = hstack([X_tfidf, X_extra])

       # get the predictions 
        pred_class = clf.predict(X_final)[0]
        pred_score = reg.predict(X_final)[0]

        # and display them
        st.success(f" Predicted Difficulty Class: **{pred_class.upper()}**")
        st.info(f" Predicted Difficulty Score: **{pred_score:.2f}**")
