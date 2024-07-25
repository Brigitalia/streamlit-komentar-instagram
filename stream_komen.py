import pickle
import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Ensure you have downloaded the stopwords
nltk.download('stopwords')

# Load the models
nb_model = pickle.load(open('komen_instagram_nb.sav', 'rb'))
svm_model = pickle.load(open('komen_instagram_svm.sav', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))

# Title of the web app
st.title('Analisis Sentimen Komentar Instagram')

# Add space between title and comment input
st.markdown("<br>", unsafe_allow_html=True)

# Input comment from user
Komentar = st.text_input('Masukkan komentar')

# Dropdown to select the model
model_option = st.selectbox('Pilih model:', ('Naive Bayes', 'SVM'))

# Preprocessing function
def preprocess_text(text):
    # Remove hashtags, mentions, and URLs
    text = re.sub(r'(@\w+|#\w+|http\S+)', '', text)
    # Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Function to create and display a word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)


# Button to check the comment
if st.button('Cek Komentar'):
    # Preprocess the user input
    cleaned_comment = preprocess_text(Komentar)
    
    # Vectorize the user input
    user_input_vector = vectorizer.transform([cleaned_comment])
    
   
 # Select the model based on user choice
    if model_option == 'Naive Bayes':
        prediction = nb_model.predict(user_input_vector)
    else:
        prediction = svm_model.predict(user_input_vector)
    
    # Display the result
    if prediction == 0:
        st.write('Sentimen: Negatif')
    elif prediction == 1:
        st.write('Sentimen: Positif')
    
    # Generate and display word cloud
    if cleaned_comment:
        st.subheader('Word Cloud dari Komentar')
        generate_word_cloud(cleaned_comment)
