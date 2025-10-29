import streamlit as st
import pandas as pd
import numpy as np
import os
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, roc_curve, precision_recall_curve, auc)
from sklearn.utils import shuffle

from collections import Counter
import altair as alt
import re

# Page config
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DATA_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
DATA_PATH = "data/sms_spam_no_header.csv"

# Styling
st.markdown("""
    <style>
    .main {
        padding: 2rem 3rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-value {
        font-size: 24px !important;
        font-weight: bold !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH, names=["label", "message"])
    df.dropna(subset=["label", "message"], inplace=True)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    df = shuffle(df, random_state=42)
    return df

@st.cache_resource
def train_model(df):
    """Train the spam classification model"""
    X = df["message"].astype(str)
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english", 
        max_df=0.9, 
        min_df=2, 
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True),
        'fpr': roc_curve(y_test, y_prob)[0],
        'tpr': roc_curve(y_test, y_prob)[1],
        'precision': precision_recall_curve(y_test, y_prob)[0],
        'recall': precision_recall_curve(y_test, y_prob)[1],
    }

    return model, vectorizer, metrics

def plot_confusion_matrix(cm):
    """Plot confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

def plot_roc_curve(fpr, tpr):
    """Plot ROC curve"""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig

def plot_class_distribution(df):
    """Plot class distribution"""
    class_counts = df['label'].value_counts().reset_index()
    class_counts.columns = ['Label', 'Count']
    
    chart = alt.Chart(class_counts).mark_bar().encode(
        x=alt.X('Label:N', title='Message Type'),
        y=alt.Y('Count:Q', title='Number of Messages'),
        color=alt.Color('Label:N', scale=alt.Scale(domain=['ham', 'spam'], 
                                                  range=['#2ecc71', '#e74c3c'])),
        tooltip=['Label', 'Count']
    ).properties(
        title='Distribution of Spam vs Ham Messages',
        width=300,
        height=200
    )
    text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        text='Count:Q'
    )
    return chart + text
def get_top_tokens(df, label, n=10):
    """Get top n tokens for a specific label"""
    texts = df[df['label'] == label]['message'].str.split()
    words = [word.lower() for text in texts for word in text if len(word) > 2]
    return Counter(words).most_common(n)
def plot_top_tokens(df):
    """Plot top tokens for both ham and spam"""
    # Get top tokens
    spam_tokens = get_top_tokens(df, 'spam', n=10)
    ham_tokens = get_top_tokens(df, 'ham', n=10)
    
    # Create dataframes
    spam_df = pd.DataFrame(spam_tokens, columns=['Token', 'Count'])
    spam_df['Type'] = 'Spam'
    ham_df = pd.DataFrame(ham_tokens, columns=['Token', 'Count'])
    ham_df['Type'] = 'Ham'
    
    # Combine dataframes
    token_df = pd.concat([spam_df, ham_df])
    
    chart = alt.Chart(token_df).mark_bar().encode(
        x=alt.X('Count:Q', title='Frequency'),
        y=alt.Y('Token:N', sort='-x', title=''),
        color=alt.Color('Type:N', 
                       scale=alt.Scale(domain=['Ham', 'Spam'], 
                                     range=['#2ecc71', '#e74c3c'])),
        tooltip=['Token', 'Count', 'Type']
    ).properties(
        width=300,
        height=300
    ).facet(
        column='Type:N',
        title='Top 10 Tokens by Message Type'
    )
    
    return chart

###
def clean_text(text: str) -> str:
    """Simple cleaning with token replacement placeholders."""
    if not isinstance(text, str):
        return ""
    # replace URLs, emails, numbers, currency, and reduce whitespace
    text = re.sub(r'(https?://\S+)|(\bwww\.\S+\b)', '<URL>', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', '<EMAIL>', text, flags=re.IGNORECASE)
    text = re.sub(r'\$\d+(\.\d+)?|\£\d+|\€\d+|\d+\s?(?:usd|eur|gbp)\b', '<CURRENCY>', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+(\.\d+)?', '<NUMBER>', text)
    text = re.sub(r'[^A-Za-z0-9\<\>\s]', ' ', text)  # remove punctuation except placeholders
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def plot_length_histogram(df):
    """Return an Altair histogram of message lengths."""
    df_len = df.assign(length=df['message'].astype(str).str.len())
    chart = alt.Chart(df_len).mark_bar().encode(
        alt.X('length:Q', bin=alt.Bin(maxbins=40), title='Message length (chars)'),
        alt.Y('count()', title='Number of messages'),
        tooltip=['count()']
    ).properties(title='Message Length Distribution', width=600, height=200)
    return chart

###

# Main app
st.title("📧 Spam Email Classifier")
st.markdown("""
    This app uses machine learning to classify emails/messages as spam or ham (non-spam).
    Try it out by entering your own message below!
""")

# Load data and train model
with st.spinner("Loading data and training model..."):
    df = load_data()
    model, vectorizer, metrics = train_model(df)

# Sidebar metrics
st.sidebar.header("Model Performance")
col1, col2 = st.sidebar.columns(2)
col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
col2.metric("F1 Score (Spam)", f"{metrics['classification_report']['spam']['f1-score']:.2%}")


###
# <-- Added: Input controls in the left panel
st.sidebar.markdown("---")
st.sidebar.header("Input Controls")
# Text size for prediction display (px)
if 'text_size' not in st.session_state:
    st.session_state['text_size'] = 18
st.session_state['text_size'] = st.sidebar.slider(
    "Text size (px)", min_value=12, max_value=36, value=st.session_state['text_size']
)

# Decision threshold for labeling as spam (0.0 - 1.0)
if 'threshold' not in st.session_state:
    st.session_state['threshold'] = 0.5
st.session_state['threshold'] = st.sidebar.slider(
    "Decision threshold (spam probability)", min_value=0.0, max_value=1.0, value=st.session_state['threshold'], step=0.01
)
###


# Main content
#tabs = st.tabs(["📝 Classifier", "📊 Model Performance", "ℹ️ About"])
tabs = st.tabs(["📝 Classifier", "📊 Model Performance", "📈 Data Overview", "ℹ️ About"])


# Classifier tab
with tabs[0]:
    st.subheader("Message Classification")

     # ensure session state key exists
    if 'message_input' not in st.session_state:
        st.session_state['message_input'] = ""

    spam_example = "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!"
    ham_example = "Hi, can we meet tomorrow at 2pm for the project discussion?"

    # Example buttons that set session state (do not rely on ephemeral local vars)
    col1, col2 = st.columns(2)
    if col1.button("Try Spam Example"):
        st.session_state['message_input'] = spam_example
    if col2.button("Try Ham Example"):
        st.session_state['message_input'] = ham_example


    # Text area bound to session state so value persists across reruns
    user_input = st.text_area(
        "Enter a message to classify:",
        key="message_input",
        height=100,
        placeholder="Type or paste your message here..."
    )



    # Classify button — user must click to send/test the message
    if st.button("Classify Message"):
        text_to_classify = st.session_state.get('message_input', "").strip()
        if not text_to_classify:
            st.warning("Please enter a message to classify.")
        else:
            # Make prediction
            X_input = vectorizer.transform([text_to_classify])
            pred = model.predict(X_input)[0]
            prob = model.predict_proba(X_input)[0]

            # Display results
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### Prediction: {'🚨 SPAM' if pred == 1 else '✅ HAM'}")
            with c2:
                st.markdown(f"### Confidence: {max(prob)*100:.1f}%")

            # Probability bar (0.0 - 1.0)
            st.progress(float(prob[1]))
            st.caption(f"Spam probability: {prob[1]:.3f}")




# Model Performance tab
with tabs[1]:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix(metrics['confusion_matrix']))
        
    with col2:
        st.subheader("ROC Curve")
        st.pyplot(plot_roc_curve(metrics['fpr'], metrics['tpr']))

# About tab
with tabs[2]:
    st.markdown("""
        ### About this Project
        This spam classifier uses:
        - **Algorithm**: Logistic Regression
        - **Features**: TF-IDF with n-grams
        - **Dataset**: SMS Spam Collection Dataset
        
        ### Technical Details
        - Training accuracy: {:.2%}
        - Spam precision: {:.2%}
        - Spam recall: {:.2%}
        
        ### Source Code
        [GitHub Repository](https://github.com/huanchen1107/2025ML-spamEmail)
    """.format(
        metrics['accuracy'],
        metrics['classification_report']['spam']['precision'],
        metrics['classification_report']['spam']['recall']
    ))

with tabs[3]:
    st.subheader("Dataset Statistics")
    
    # Display basic statistics
    total_messages = len(df)
    spam_messages = len(df[df['label'] == 'spam'])
    ham_messages = len(df[df['label'] == 'ham'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Messages", f"{total_messages:,}")
    col2.metric("Spam Messages", f"{spam_messages:,}")
    col3.metric("Ham Messages", f"{ham_messages:,}")
    
    st.subheader("Class Distribution")
    distribution_chart = plot_class_distribution(df)
    st.altair_chart(distribution_chart, use_container_width=True)
    
    st.subheader("Message Lengths")
    st.altair_chart(plot_length_histogram(df), use_container_width=True)

    ##st.subheader("Most Common Words (Top 10 per class)")
    ##tokens_chart = plot_top_tokens(df)
    ##st.altair_chart(tokens_chart, use_container_width=True)

    st.markdown("#### Top tokens — tables")
    # create tables of top tokens for ham/spam
    spam_top = get_top_tokens(df, 'spam', n=20)
    ham_top = get_top_tokens(df, 'ham', n=20)
    spam_df = pd.DataFrame(spam_top, columns=['token', 'count'])
    ham_df = pd.DataFrame(ham_top, columns=['token', 'count'])
    t1, t2 = st.columns(2)

    st.subheader("Most Common Words")
    tokens_chart = plot_top_tokens(df)
    st.altair_chart(tokens_chart, use_container_width=True)
    ## Sample messages
    st.subheader("Sample Messages")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Sample Spam Messages")
        spam_samples = df[df['label'] == 'spam']['message'].sample(3, random_state=42)
        for i, msg in enumerate(spam_samples, 1):
            st.markdown(f"{i}. _{msg[:100]}{'...' if len(msg) > 100 else ''}_")
    
    with col2:
        st.markdown("##### Sample Ham Messages")
        ham_samples = df[df['label'] == 'ham']['message'].sample(3, random_state=42)
        for i, msg in enumerate(ham_samples, 1):
            st.markdown(f"{i}. _{msg[:100]}{'...' if len(msg) > 100 else ''}_")


# Footer
st.markdown("---")
st.caption("© 2025 AIoT Course Demo | Built with Streamlit")