import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
def load_test_data():
        df = pd.read_csv(r'c:\Users\DELL\Downloads\HateSpeechDataset.csv\HateSpeechDataset.csv')
        X = df['Content']
        y = df['Label']
        return X, y

# Evaluate model
def evaluate_model():
    try:
        model = joblib.load('hate_speech_model.joblib')
    except:
        print("Model file not found. Please train the model first.")
        return
    
    X, y = load_test_data()
    if X is None:
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    # Evaluate
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    evaluate_model()