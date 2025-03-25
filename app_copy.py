import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess and transform text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]

    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    return " ".join(y)

# Load the vectorizer and model
tf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app layout
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📧", layout="centered")

st.title("📊 Email/SMS Spam Classifier")
st.write("Classify your messages in real-time as it is Spam or Not Spam.")

# Add a sidebar with additional info
st.sidebar.header("📌 About the App")
st.sidebar.write("This app uses Machine Learning Algorithm and Natural Language Processing (NLP) techniques to classify whether an input message is spam or not spam.")
st.sidebar.write("Model: Naive Bayes Classifier")
st.sidebar.write("Vectorization: TF-IDF Vectorizer")

# Input box for message
st.markdown("### 📩 Enter the Message Below :-")
input_sms = st.text_area("", placeholder="Type your message here...", height=150)

# Predict button with custom styling
if st.button('🚀 Predict', use_container_width=True):
    if input_sms:
        # Preprocess the message
        transformed_sms = transform_text(input_sms)

        # Vectorize the message
        vector_input = tf.transform([transformed_sms])

        # Predict using the model
        result = model.predict(vector_input)[0]

        # Display the result with colored banners
        if result == 1:
            st.markdown("""<div style='text-align: center; padding: 20px; border-radius: 10px; background-color: #FF4B4B; color: white; font-size: 24px;'>🚨 Spam</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style='text-align: center; padding: 20px; border-radius: 10px; background-color: #4CAF50; color: white; font-size: 24px;'>✅ Not Spam</div>""", unsafe_allow_html=True)

        # Display transformed text for debugging or understanding
        #st.markdown("**🔍 Preprocessed Message:**")
        #st.code(transformed_sms)

    else:
        st.warning("⚠️ Please enter a message to classify.")

"""
# Footer
st.markdown("""
---
💡 **Note:** This model is trained for educational purposes. Always verify with real-world testing for production use.
""")
"""
