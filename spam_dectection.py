import pandas as pd
from pathlib import Path
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------- Load CSV --------
BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "spam.csv"

if not CSV_PATH.exists():
    st.error(f"CSV not found at: {CSV_PATH}")
    st.stop()

data = pd.read_csv(CSV_PATH, encoding="latin-1")
data = data.dropna(axis=1, how="all").drop_duplicates()
data.columns = data.columns.str.strip()

# Rename columns if needed
rename_map = {"v1": "Category", "v2": "Message", "label": "Category", "text": "Message"}
for k, v in rename_map.items():
    if k in data.columns and v not in data.columns:
        data = data.rename(columns={k: v})

if "Category" not in data.columns or "Message" not in data.columns:
    st.error(f"Expected 'Category' and 'Message'. Found: {list(data.columns)}")
    st.stop()

# Clean labels
data["Category"] = (
    data["Category"].astype(str).str.strip().str.lower()
    .replace({"ham": "Not Spam", "spam": "Spam", "0": "Not Spam", "1": "Spam"})
)

# -------- Train model --------
X = data["Message"].astype(str)
y = data["Category"].astype(str)

cv = CountVectorizer(stop_words="english")
X_vec = cv.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# -------- UI --------
st.title("📩 SMS Spam Detection")
st.caption("CountVectorizer + MultinomialNB")

msg = st.text_area("Type an SMS to classify:", height=140, placeholder="e.g., Congratulations! You won a prize...")

if st.button("Classify"):
    if msg.strip():
        pred = model.predict(cv.transform([msg]))[0]
        st.success(f"Prediction: **{pred}**")
    else:
        st.warning("Please enter a message first.")
