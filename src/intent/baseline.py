"""
Baseline Intent Classifiers.
- Majority class baseline
- TF-IDF + Logistic Regression
Run: python -m src.intent.baseline
"""
import logging
import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

log = logging.getLogger(__name__)


class MajorityClassifier:
    """Simple majority class baseline."""
    
    def __init__(self):
        self.majority_class = None
    
    def fit(self, X, y):
        """Fit by finding the most common class."""
        self.majority_class = pd.Series(y).mode()[0]
        return self
    
    def predict(self, X):
        """Always predict the majority class."""
        return [self.majority_class] * len(X)
    
    def predict_proba(self, X):
        """Return probability (1.0 for majority class)."""
        return np.ones((len(X), 1))


class KeywordClassifier:
    """Keyword-based intent classifier using the taxonomy."""
    
    def __init__(self):
        self.intent_keywords = {
            'late_delivery': ['late', 'delayed', 'delay', 'not arrived', 'hasn\'t arrived', 'where is', 'when will', 'overdue', 'still waiting', 'expected delivery', 'delivery date'],
            'missing_package': ['missing', 'never received', 'not received', 'didn\'t receive', 'lost', 'stolen', 'not delivered', 'package missing'],
            'wrong_item': ['wrong item', 'wrong product', 'incorrect item', 'received wrong', 'different item', 'not what i ordered'],
            'refund_issue': ['refund', 'money back', 'reimburse', 'credited back', 'refund status'],
            'payment_issue': ['charged twice', 'double charge', 'overcharged', 'payment', 'billing', 'transaction', 'payment failed'],
            'order_cancellation': ['cancel', 'cancellation', 'cancel order', 'stop order'],
            'return_item': ['return', 'send back', 'returning', 'return label', 'return shipping'],
            'account_issue': ['account', 'password', 'login', 'sign in', 'can\'t access', 'locked out', 'suspended'],
            'technical_issue': ['app', 'website', 'not working', 'error', 'bug', 'crash', 'technical', 'loading'],
            'product_inquiry': ['out of stock', 'available', 'restock', 'in stock', 'availability'],
            'gift_card': ['gift card', 'gift certificate', 'redeem', 'balance'],
            'promotion': ['promo', 'coupon', 'discount', 'deal', 'offer', 'promotion', 'code'],
            'complaint': ['unacceptable', 'terrible', 'worst', 'disgusted', 'furious', 'angry', 'frustrated'],
            'general_inquiry': ['question', 'help', 'info', 'how do i', 'how to', 'can i'],
        }
    
    def fit(self, X, y=None):
        """No fitting needed for keyword classifier."""
        return self
    
    def predict(self, X):
        """Classify using keywords."""
        predictions = []
        for text in X:
            if not isinstance(text, str):
                predictions.append('general_inquiry')
                continue
            
            text_lower = text.lower()
            scores = {}
            for intent, keywords in self.intent_keywords.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                if score > 0:
                    scores[intent] = score
            
            if scores:
                predictions.append(max(scores, key=scores.get))
            else:
                predictions.append('general_inquiry')
        
        return predictions


class TFIDFClassifier:
    """TF-IDF + Logistic Regression classifier."""
    
    def __init__(self, max_features=10000, C=1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the TF-IDF vectorizer and classifier."""
        X_tfidf = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_tfidf, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict intents."""
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict(X_tfidf)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_tfidf)
    
    def save(self, path):
        """Save the model."""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
    
    def load(self, path):
        """Load the model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.is_fitted = True


def train_and_evaluate(classifier, X_train, y_train, X_test, y_test, name):
    """Train and evaluate a classifier."""
    log.info("\n%s", "=" * 60)
    log.info("%s", name)
    log.info("%s", "=" * 60)

    # Train
    classifier.fit(X_train, y_train)

    # Predict
    y_pred = classifier.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    log.info("\nAccuracy: %.4f", accuracy)
    log.info("Macro F1: %.4f", f1_macro)
    log.info("Weighted F1: %.4f", f1_weighted)

    log.info("\nClassification Report:")
    log.info("%s", classification_report(y_test, y_pred, zero_division=0))
    
    return {
        'accuracy': accuracy,
        'macro_f1': f1_macro,
        'weighted_f1': f1_weighted,
        'predictions': y_pred
    }


def main():
    """Train and evaluate baseline classifiers."""
    # Load data
    data_path = "data/processed/filtered_conversations.csv"
    log.info("Loading data from %s...", data_path)
    df = pd.read_csv(data_path, low_memory=False)
    log.info("Loaded %s conversations", f"{len(df):,}")

    # Prepare data
    X = df['customer_problem'].fillna('').astype(str).values
    y = df['intent'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info("\nTrain: %s, Test: %s", f"{len(X_train):,}", f"{len(X_test):,}")
    
    # Results storage
    results = {}
    
    # 1. Majority class baseline
    majority = MajorityClassifier()
    results['majority'] = train_and_evaluate(
        majority, X_train, y_train, X_test, y_test, "Majority Class Baseline"
    )
    
    # 2. Keyword classifier
    keyword = KeywordClassifier()
    results['keyword'] = train_and_evaluate(
        keyword, X_train, y_train, X_test, y_test, "Keyword Classifier"
    )
    
    # 3. TF-IDF + Logistic Regression
    tfidf = TFIDFClassifier()
    results['tfidf_lr'] = train_and_evaluate(
        tfidf, X_train, y_train, X_test, y_test, "TF-IDF + Logistic Regression"
    )
    
    # Save TF-IDF model
    model_dir = "artifacts/intent_model"
    os.makedirs(model_dir, exist_ok=True)
    tfidf.save(os.path.join(model_dir, "tfidf_lr.pkl"))
    log.info("\nSaved TF-IDF model to %s/tfidf_lr.pkl", model_dir)

    # Save results
    results_path = "artifacts/intent_model/baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'predictions'}
                   for k, v in results.items()}, f, indent=2)
    log.info("Saved results to %s", results_path)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
