import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
nltk.download('stopwords')

# Preprocessing
def preprocess_text(text):
    # Tokenisasi
    tokens = nltk.word_tokenize(text)

    # Konversi ke huruf kecil
    tokens = [token.lower() for token in tokens]

    # Penghapusan kata-kata stopword
    stop_words = set(stopwords.words('indonesian'))
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
tfidf_vect_8020 = joblib.load("vectorizer.pkl")

def classify_review(review):
    # Preprocess the input text
    review = " ".join(preprocess_text(review))
    review = tfidf_vect_8020.transform([review])
    # Make predictions
    prediction = model.predict(review)
    return prediction[0]

# Create a Streamlit web app
def main():
    st.title("Review Classification")
    st.write("Masukan sebuah review dan klik tombol 'Classify' untuk memprediksi label-nya (postive atau negatif).")
    
    
    # Input text box
    review = st.text_area("Masukkan sebuah review", "")
    
    # Classify button
    if st.button("Classify"):
        if review:
            # Perform classification
            prediction = "Positif" if classify_review(review) == 1 else "Negatif"
            st.write("Prediksi:", prediction)
        else:
            st.write("Silakan masukkan sebuah review.")

if __name__ == "__main__":
    main()
