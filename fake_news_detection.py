# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 2. Load the Fake and Real news datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# 3. Add labels manually
df_fake["label"] = 1   # FAKE
df_true["label"] = 0   # REAL

# 4. Combine both datasets
df = pd.concat([df_fake, df_true])
df = df[['title', 'text', 'label']]  # keep only relevant columns
df.dropna(inplace=True)

# 5. Create a combined text column
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]]  # final usable dataframe

# 6. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# 7. Vectorize the text using TF-IDF
tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 8. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 9. Make predictions and evaluate
y_pred = model.predict(X_test_vec)
print("\n✅ Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 10. Save the model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\n✅ Model and Vectorizer saved as .pkl files.")
