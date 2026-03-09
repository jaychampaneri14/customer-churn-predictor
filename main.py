"""
Customer Churn Predictor — XGBoost with SHAP explanations
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib, warnings
warnings.filterwarnings("ignore")

def generate_churn_data(n=5000, seed=42):
    """Generate synthetic telecom churn data."""
    np.random.seed(seed)
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "monthly_charges": np.random.uniform(20, 120, n),
        "total_charges": np.random.uniform(100, 8000, n),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.5, 0.3, 0.2]),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "payment_method": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "num_support_calls": np.random.poisson(1.5, n),
        "num_products": np.random.randint(1, 6, n),
        "satisfaction_score": np.random.randint(1, 6, n),
        "age": np.random.randint(18, 80, n),
    })
    # Churn probability based on features
    churn_prob = (
        0.3 * (df["contract"] == "Month-to-month").astype(float) +
        0.2 * (df["tenure"] < 12).astype(float) +
        0.2 * (df["satisfaction_score"] < 3).astype(float) +
        0.1 * (df["num_support_calls"] > 3).astype(float) +
        np.random.normal(0, 0.1, n)
    ).clip(0, 1)
    df["churn"] = (churn_prob > 0.35).astype(int)
    return df

def preprocess(df):
    le = LabelEncoder()
    for col in ["contract", "internet_service", "payment_method"]:
        df[col] = le.fit_transform(df[col])
    X = df.drop("churn", axis=1)
    y = df["churn"]
    return X, y

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                              use_label_encoder=False, eval_metric="logloss", random_state=42)
        print("Using XGBoost")
    except ImportError:
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        print("Using Random Forest (install xgboost for better results)")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    # Feature importance
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop Churn Factors:")
    print(fi.head(5).to_string())

    joblib.dump(model, "churn_model.pkl")
    print("\nModel saved: churn_model.pkl")

    # SHAP explanations
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test[:100])
        shap.summary_plot(shap_vals, X_test[:100], show=False)
        plt.savefig("shap_summary.png", dpi=100, bbox_inches="tight")
        print("SHAP plot saved: shap_summary.png")
    except ImportError:
        print("Install shap for model explanations: pip install shap")

    return model

def main():
    print("=" * 60)
    print("  Customer Churn Predictor")
    print("=" * 60)
    print("\nGenerating synthetic customer data...")
    df = generate_churn_data()
    print(f"Dataset: {len(df)} customers, churn rate: {df.churn.mean():.1%}")
    X, y = preprocess(df)
    print("\nTraining model...")
    model = train(X, y)

if __name__ == "__main__":
    main()
