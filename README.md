# SMS-Email-Spam-Detection

## Description
This project is a spam detection system for SMS and email messages. It classifies text as "spam" or "not spam" using machine learning and natural language processing (NLP) techniques. The pipeline involves collecting a dataset from Kaggle, performing exploratory data analysis (EDA), data cleaning, and text preprocessing in Jupyter Notebook, followed by training multiple machine learning models. The best-performing model (Naive Bayes) was selected based on precision and accuracy, pickled, and deployed as an interactive web app using Streamlit.

## Required Tools and Libraries
- **Python 3.8+** - Core language for data processing and model building
- **Jupyter Notebook** - For data exploration, cleaning, and model training
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning models, CountVectorizer, and TfidfVectorizer
- **NLTK** - NLP text preprocessing (e.g., tokenization, stop words removal)
- **Streamlit** - Web app framework for deployment
- **Pickle** - Model serialization for saving and loading the trained Naive Bayes model

## Features
- Data exploration and visualization for insights into SMS/email spam patterns
- NLP-based text preprocessing (e.g., tokenization, stemming, stop words removal)
- Comparison of multiple ML models (e.g., Naive Bayes, Logistic Regression, etc.) with CountVectorizer and TfidfVectorizer
- Deployment of the highest-performing Naive Bayes model as a Streamlit web app
- Real-time spam detection: input a message and get instant classification

## Installation
1. Clone the repository: `git clone https://github.com/Ashutosh-jarag/Sms-Email-spam-Detection.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Download the dataset from Kaggle: ['https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset']
4. Place the dataset (e.g., `spam.csv`) in the project’s root directory
5. Run the Jupyter Notebook (`Spam_Detection.ipynb`) for data processing and model training
6. Deploy the web app: `streamlit run app.py`

## Usage
1. Open the Jupyter Notebook (`Spam_Detection.ipynb`) to:
   - Perform EDA, data cleaning, and transformation
   - Train ML models and pickle the Naive Bayes model
2. Launch the Streamlit app:
   - Run `streamlit run app.py` in your terminal
   - Open your browser at `http://localhost:8501`
   - Enter a message in the text box and click "Predict" to see if it’s spam or not

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
MIT License
