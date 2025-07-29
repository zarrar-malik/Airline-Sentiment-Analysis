import os
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


# Get the directory where your script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root, then into data folder
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "data", "Tweets.csv")
df = pd.read_csv(csv_path)
#df = pd.read_csv("Tweets.csv")  # Update with your path
df = df[['text', 'airline_sentiment', 'airline_sentiment_confidence']].dropna()
df = df[df['airline_sentiment_confidence'] > 0.7]

# === Step 2: Enhanced Text Preprocessing ===
def enhanced_clean_text(text):
    # Basic cleaning
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    
    # Advanced cleaning
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['text'].apply(enhanced_clean_text)

# === Step 3: Encode Labels ===
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['airline_sentiment'])

# === Step 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['sentiment_encoded'],
    test_size=0.2,
    stratify=df['sentiment_encoded'],
    random_state=42
)

# === Step 5: Define Models ===
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# === Step 6: Train and Evaluate Models ===
results = {}
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    
    # Create pipeline with SMOTE
    pipeline = make_pipeline(
        TfidfVectorizer(stop_words='english'),
        SMOTE(random_state=42),
        model
    )
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    
    # Store results
    results[name] = {
        'model': pipeline,
        'report': report,
        'accuracy': report['accuracy']
    }
    
    # Print results
    print(f"Accuracy: {report['accuracy']:.2f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

# === Step 7: Select Best Model ===
best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.2f}")

# === Step 8: Prediction Function ===
def predict_sentiment(text, model=best_model):
    cleaned = enhanced_clean_text(text)
    encoded_pred = model.predict([cleaned])[0]
    return label_encoder.inverse_transform([encoded_pred])[0]

# === Example Predictions ===
examples = [
    "Thanks for checking.  Please go ahead and close the ticket.",
    "Sounds good, thank you!",
    "This information answers my question. Thank you so much!",
    
    "My flight was delayed for 4 hours and no one helped.",
    "Amazing service by JetBlue today!",
    "It was okay. Nothing special.",
    "Absolutely the worst experience I've had on a plane.",
    "Crew was friendly and everything was on time."
]

print("\n=== Example Predictions ===")
for tweet in examples:
    print(f"{tweet} --> {predict_sentiment(tweet)}")