# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keybert import KeyBERT
from transformers import pipeline
import nltk
nltk.download('vader_lexicon')

#
# Functions

def clean_text(df, text_column='employee_feedback'):
    df['clean_text'] = (
        df[text_column].astype(str)
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.strip()
    )
    return df

def fit_nmf_model(df, text_column='clean_text', n_topics=10):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df[text_column])
    nmf_model = NMF(n_components=n_topics, random_state=42)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_
    feature_names = vectorizer.get_feature_names_out()
    return nmf_model, W, H, feature_names

def assign_topics(df, W):
    df['topic'] = W.argmax(axis=1)
    return df

def get_topic_words(H, feature_names, n_words=10):
    topic_words = {}
    for i, topic in enumerate(H):
        top_words = [feature_names[j] for j in topic.argsort()[-n_words:][::-1]]
        topic_words[i] = top_words
    return topic_words

def generate_topic_labels(topic_words):
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    topic_labels = {}
    for t, words in topic_words.items():
        topic_text = " ".join(words)
        keywords = kw_model.extract_keywords(topic_text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=1)
        label = keywords[0][0] if keywords else f"Topic {t}"
        topic_labels[t] = label
    return topic_labels

def employee_sentiment(df, text_column='clean_text'):
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
    )
    return df

def department_topic_summary(df):
    summary = df.groupby(['department','topic']).size().reset_index(name='count')
    return summary

def plot_department_topics_with_labels(df, topic_label_col='topic_label'):
    """
    Plot department-wise topic distribution with topic labels on each bar.
    """
    summary = df.groupby(['department', topic_label_col]).size().reset_index(name='count')
    
    plt.figure(figsize=(12,6))
    ax = sns.barplot(
        data=summary,
        x='department',
        y='count',
        hue=topic_label_col
    )
    
    # Add labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{p.get_height():.0f}',
                (p.get_x() + p.get_width() / 2., height),
                ha='center',
                va='bottom',
                fontsize=9,
                rotation=0
            )
    
    plt.title("Department-wise Topic Distribution with Labels")
    plt.ylabel("Number of Employees")
    plt.xlabel("Department")
    plt.xticks(rotation=45)
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt.gcf())

def cluster_employee_profiles(df, W, n_clusters=5):
    features = np.hstack([W, df['sentiment_score'].values.reshape(-1,1)])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['employee_profile'] = kmeans.fit_predict(features_scaled)
    return df, kmeans

def list_employees_by_cluster_with_topic_and_sentiment(
    df, 
    cluster_col='employee_profile', 
    id_col='unique_identifier', 
    dept_col='department', 
    feedback_col='employee_feedback', 
    topic_col='topic', 
    topic_label_col='topic_label',
    sentiment_col='sentiment_score',
    role_col='inferred_role'
):
    cluster_dict = {}
    clusters = df[cluster_col].unique()
    for c in clusters:
        cluster_df = df[df[cluster_col] == c].copy()
        avg_sent = cluster_df[sentiment_col].mean()
        if avg_sent >= 0.05:
            sentiment_label = "Mostly Positive"
        elif avg_sent <= -0.05:
            sentiment_label = "Mostly Negative"
        else:
            sentiment_label = "Neutral"
        cluster_name = f"Cluster {c} – {sentiment_label} (avg={avg_sent:.2f})"
        columns_to_show = [id_col, dept_col, feedback_col, topic_col, topic_label_col, sentiment_col, role_col]
        cluster_dict[cluster_name] = cluster_df[columns_to_show].reset_index(drop=True)
    return cluster_dict

def visualize_employee_clusters_3d(df, W, cluster_col='employee_profile', n_components=3):
    features = np.hstack([W, df['sentiment_score'].values.reshape(-1,1)])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(features_scaled)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    clusters = df[cluster_col].unique()
    colors = plt.cm.tab10.colors
    for i, c in enumerate(clusters):
        idx = df[cluster_col] == c
        ax.scatter(
            components[idx,0], components[idx,1], components[idx,2],
            c=[colors[i%10]], label=f'Cluster {c}',
            s=50, alpha=0.7
        )
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.set_title('Employee Clusters by Topic Distribution + Sentiment')
    ax.legend()
    st.pyplot(fig)

def infer_roles(df, text_column='employee_feedback'):
    st.info("Inferring roles...")
    candidate_roles = [
        "Entry-level",
        "Early Career",
        "Mid-level",
        "Lower Management",
        "Upper Management",
        "Executive / C-suite"
    ]
    role_classifier = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli"
    )
    df['inferred_role'] = df[text_column].apply(
        lambda x: "unknown" if not isinstance(x,str) or len(x.strip())==0 else
                  role_classifier(x, candidate_labels=candidate_roles, multi_label=False)['labels'][0]
    )
    return df

def plot_roles_by_department(df):
    st.subheader("Inferred Roles by Department (Objective 3, can sense check against known distribution but not foolproof)")
    role_counts = df.groupby(['department','inferred_role']).size().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(data=role_counts, x='department', y='count', hue='inferred_role', ax=ax)
    plt.xticks(rotation=45)
    plt.ylabel("Number of Employees")
    plt.xlabel("Department")
    plt.title("Role Distribution by Department")
    plt.legend(title="Role", bbox_to_anchor=(1.05,1), loc='upper left')
    st.pyplot(fig)


# Streamlit App

st.title("Employee Feedback Topic, Sentiment & Role Analyzer")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv','xls','xlsx'])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('xls','xlsx')) else pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())

    df = clean_text(df)

    n_topics = st.slider("Number of Topics", 5, 20, 10)
    nmf_model, W, H, feature_names = fit_nmf_model(df, n_topics=n_topics)
    df = assign_topics(df, W)
    topic_words = get_topic_words(H, feature_names)
    topic_labels = generate_topic_labels(topic_words)
    df['topic_label'] = df['topic'].map(topic_labels)


    df = employee_sentiment(df)

    st.subheader("Topic Summary (Objective 1)")
    sia = SentimentIntensityAnalyzer()
    topic_sentiments = {t: sia.polarity_scores(" ".join(words))['compound'] for t, words in topic_words.items()}
    topic_summary_df = pd.DataFrame({
        'topic': list(topic_words.keys()),
        'top_words': [" ".join(w) for w in topic_words.values()],
        'label': [topic_labels[t] for t in topic_words.keys()],
        'sentiment': [("Positive" if topic_sentiments[t]>=0.05 else "Negative" if topic_sentiments[t]<=-0.05 else "Neutral") 
                      for t in topic_words.keys()]
    })
    st.dataframe(topic_summary_df)


    st.subheader("Department-wise Topic Distribution (Objective 2)")
    dept_summary = department_topic_summary(df)
    plot_department_topics_with_labels(df, topic_label_col='topic_label')

 
    df = infer_roles(df)
    


    st.subheader("Employee Feedback with Inferred Roles (Objective 3)")
    st.dataframe(df[[ 'inferred_role','unique_identifier', 'department', 'employee_feedback']])

    plot_roles_by_department(df)