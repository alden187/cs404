"""
Fake News Detection using a Shallow Neural Network
AI Class Project

Uses TF-IDF features + scikit-learn MLPClassifier (1 hidden layer)
Dataset: Kaggle Fake and Real News dataset (Fake.csv + True.csv)
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ── Step 1: Load & Clean Data ────────────────────────────────────────────────

fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

fake_df["label"] = 1  # 1 = fake
true_df["label"] = 0  # 0 = real

df = pd.concat([true_df, fake_df], ignore_index=True)

# Drop rows where text is missing
df = df.dropna(subset=["text"])

# Combine title and text into a single feature
df["content"] = df["title"].fillna("") + " " + df["text"]

X = df["content"]
y = df["label"]

print(f"Dataset: {len(df)} articles ({(y == 0).sum()} real, {(y == 1).sum()} fake)\n")


# ── Step 2: Text Preprocessing with TF-IDF ───────────────────────────────────

def clean_text(text):
    """Lowercase, remove punctuation and numbers."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

X = X.apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_tfidf = vectorizer.fit_transform(X)

print(f"TF-IDF feature matrix: {X_tfidf.shape}\n")


# ── Step 3: Train/Test Split ─────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)


# ── Step 4: Shallow Neural Network ───────────────────────────────────────────

print("=" * 60)
print("SHALLOW NEURAL NETWORK (MLPClassifier — 1 hidden layer, 64 neurons)")
print("=" * 60)

mlp = MLPClassifier(
    hidden_layer_sizes=(64,),  # 1 hidden layer with 64 neurons
    activation="relu",
    max_iter=20,
    random_state=42,
    verbose=True,
)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred_mlp, target_names=["Real", "Fake"]))


# ── Step 5: Baseline Comparisons ─────────────────────────────────────────────

print("=" * 60)
print("BASELINE: Logistic Regression")
print("=" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(classification_report(y_test, y_pred_lr, target_names=["Real", "Fake"]))

print("=" * 60)
print("BASELINE: Naive Bayes")
print("=" * 60)

nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
print(classification_report(y_test, y_pred_nb, target_names=["Real", "Fake"]))


# ── Step 6: Confusion Matrix Plot ────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, y_pred, title in zip(
    axes,
    [y_pred_mlp, y_pred_lr, y_pred_nb],
    ["Shallow Neural Network", "Logistic Regression", "Naive Bayes"],
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()

print("\nConfusion matrix plot saved to confusion_matrices.png")


# ── Step 7: Interactive Prediction ───────────────────────────────────────────

def predict_article(text):
    """Predict whether a given article text is real or fake."""
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = mlp.predict(features)[0]
    confidence = mlp.predict_proba(features)[0]
    label = "FAKE" if prediction == 1 else "REAL"
    pct = confidence[prediction] * 100
    print(f"\n  Prediction: {label} (confidence: {pct:.1f}%)\n")


print("\n" + "=" * 60)
print("INTERACTIVE MODE — Paste an article to check if it's fake")
print("Type 'quit' to exit")
print("=" * 60)

while True:
    print()
    article = input("Paste article text: ").strip()
    if article.lower() in ("quit", "exit", "q"):
        break
    if not article:
        print("  Please paste some text.")
        continue
    predict_article(article)
