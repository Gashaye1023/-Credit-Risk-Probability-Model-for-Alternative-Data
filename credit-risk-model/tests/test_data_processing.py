import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
df_main=pd.read_csv('Final_processed_data.csv')
X = df_main.drop(columns=['is_high_risk'])
y = df_main['is_high_risk']

    # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
print(f"Distribution of target in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Distribution of target in testing set:\n{y_test.value_counts(normalize=True)}")


# Function to train, evaluate, and log model with MLflow
def train_evaluate_log_model(model, model_name, X_train, y_train, X_test, y_test, params=None):
    with mlflow.start_run(run_name=model_name) as run:
        # Log parameters
        mlflow.log_param("model_name", model_name)
        if params:
            mlflow.log_params(params)
        else:
            mlflow.log_params(model.get_params())

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] # Probability for the positive class

        # Evaluate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"\n--- {model_name} Metrics ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Log the model
        mlflow.sklearn.log_model(model, "model")
        print(f"Model '{model_name}' logged to MLflow.")
        return roc_auc, run.info.run_id # Return ROC_AUC and run_id for best model selection


# --- Choose and Train Models ---

best_roc_auc = -1
best_model_name = ""
best_run_id = ""

# Model 1: Logistic Regression (with Hyperparameter Tuning)
print("\n--- Training Logistic Regression with GridSearchCV ---")
log_reg = LogisticRegression(random_state=random_state, solver='liblinear')
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2']
}
grid_search_lr = GridSearchCV(log_reg, param_grid_lr, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search_lr.fit(X_train, y_train)

best_lr_model = grid_search_lr.best_estimator_
print(f"Best Logistic Regression parameters: {grid_search_lr.best_params_}")
current_roc_auc, current_run_id = train_evaluate_log_model(best_lr_model, "Logistic Regression (Tuned)", X_train, y_train, X_test, y_test, params=grid_search_lr.best_params_)

if current_roc_auc > best_roc_auc:
    best_roc_auc = current_roc_auc
    best_model_name = "Logistic Regression (Tuned)"
    best_run_id = current_run_id


# Model 2: Random Forest Classifier
print("\n--- Training Random Forest Classifier ---")
rf_model = RandomForestClassifier(random_state=random_state, n_estimators=100)
current_roc_auc, current_run_id = train_evaluate_log_model(rf_model, "Random Forest", X_train, y_train, X_test, y_test)

if current_roc_auc > best_roc_auc:
    best_roc_auc = current_roc_auc
    best_model_name = "Random Forest"
    best_run_id = current_run_id

print(f"\n--- Best Model Identified ---")
print(f"Best Model: {best_model_name}")
print(f"Best ROC AUC: {best_roc_auc:.4f}")
print(f"Best Run ID: {best_run_id}")

# --- Register the Best Model in MLflow Model Registry ---
if best_run_id:
    print(f"\nRegistering '{best_model_name}' (Run ID: {best_run_id}) to MLflow Model Registry...")
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name="CreditRiskHighRiskModel")
    print(f"Model registered as: {registered_model.name} (Version: {registered_model.version})")
else:
    print("\nNo best model identified or run ID available for registration.")

print("\n--- Model Training, Tracking, and Registration Complete ---")
print("You can view the MLflow runs and registered models by starting the MLflow UI in your environment.")