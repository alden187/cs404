"""
Train the fake news detection model and save it to disk.
Run this once before launching the GUI.
"""

import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report


def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def main():
    print("Loading data...")
    fake_df = pd.read_csv("data/Fake.csv")
    true_df = pd.read_csv("data/True.csv")

    fake_df["label"] = 1
    true_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["content"] = df["title"].fillna("") + " " + df["text"]

    X = df["content"].apply(clean_text)
    y = df["label"]

    print(f"Dataset: {len(df)} articles ({(y == 0).sum()} real, {(y == 1).sum()} fake)")

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

    print("Training neural network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        max_iter=20,
        random_state=42,
        verbose=True,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    print("Saving model to model.pkl...")
    with open("model.pkl", "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": mlp}, f)

    print("Done! You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
