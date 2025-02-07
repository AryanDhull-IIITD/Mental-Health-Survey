import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
import shap
import joblib
import xgboost as xgb
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

"""DATASET LOADING AND DESCRIPTION"""

# Load Dataset
data_path = "survey.csv"
df = pd.read_csv(data_path)

# Dataset Information and Initial Exploration
print("Dataset Info:")
df.info()

print("\nFirst 5 Rows:")
df.head()

print("\nDescriptive Statistics:")
df.describe()

print("\nMissing Values Count:")
print(df.isnull().sum())

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\nDataset After Handling Missing Values:")
df.info()

"""EDA"""

# Data Visualization
sns.set(style="whitegrid")

# Filter age to a reasonable range
df = df[(df["Age"] >= 10) & (df["Age"] <= 100)]

plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True, color="skyblue")
plt.title("Age Distribution (Cleaned)")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 12))
sns.countplot(y=df["Gender"], order=df["Gender"].value_counts().index, palette="coolwarm")
plt.title("Gender Distribution")
plt.xlabel("Count")
plt.ylabel("Gender")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=df["self_employed"], palette="viridis")
plt.title("Employment Type Distribution")
plt.xlabel("Self-Employed")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x=df["treatment"], palette="Set2")
plt.title("Mental Health Treatment Seeking")
plt.xlabel("Sought Treatment (Yes/No)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x="treatment", y="Age", hue="Gender", data=df, palette="coolwarm")
plt.title("Mental Health Treatment by Age & Gender")
plt.xlabel("Sought Treatment (Yes/No)")
plt.ylabel("Age")
plt.show()

"""FEATURE ENGINEERING AND HANDLING OUTLIERS"""

# Feature Engineering
df["treatment_numeric"] = df["treatment"].map({"Yes": 1, "No": 0})
df["self_employed_numeric"] = df["self_employed"].map({"Yes": 1, "No": 0})
df["family_history_numeric"] = df["family_history"].map({"Yes": 1, "No": 0})

numerical_cols = df.select_dtypes(include=["number"]).columns  # Get numerical columns
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="Blues", fmt=".2f")
plt.title("Feature Correlation Heatmap (Fixed)")
plt.show()

df["age_group"] = pd.cut(df["Age"], bins=[10, 25, 40, 60, 100], labels=[0, 1, 2, 3])

support_cols = ["benefits", "care_options", "wellness_program", "seek_help"]
df["company_support_score"] = df[support_cols].apply(lambda x: np.sum(x == "Yes"), axis=1)

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

age_outliers = detect_outliers_iqr(df, "Age")
print(f"Outliers in Age: {len(age_outliers)}")

df["age_zscore"] = np.abs(stats.zscore(df["Age"]))
zscore_outliers = df[df["age_zscore"] > 3]  # Z-score > 3 is considered an outlier
print(f"Outliers in Age (Z-score method): {len(zscore_outliers)}")

age_cap_upper = df["Age"].quantile(0.95)
age_cap_lower = df["Age"].quantile(0.05)
df["Age"] = np.clip(df["Age"], age_cap_lower, age_cap_upper)

"""FEATURE SELECTION AND HANDLING CLASS IMBALANCE"""

features = [
    "self_employed_numeric", "family_history_numeric", "age_group",
    "company_support_score", "work_interfere", "no_employees", "remote_work", "tech_company",
    "anonymity", "leave", "mental_health_consequence", "phys_health_consequence",
    "coworkers", "supervisor", "mental_health_interview", "phys_health_interview",
    "mental_vs_physical", "obs_consequence"
]

target = "treatment_numeric"

df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.select_dtypes(include=['object']).columns:
    if col in features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

X = df_encoded[features]
y = df_encoded[target]

# Handle class imbalance by undersampling the majority class
count_class_0, count_class_1 = y.value_counts()

df_class_0 = df_encoded[df_encoded[target] == 0]
df_class_1 = df_encoded[df_encoded[target] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_under = pd.concat([df_class_0_under, df_class_1], axis=0)

X = df_under[features]
y = df_under[target]

"""MODELLING"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

"""1)Random Forest"""

rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_

# Evaluate Random Forest Model
y_pred_rf_best = best_rf.predict(X_test)
y_pred_proba_rf_best = best_rf.predict_proba(X_test)[:, 1]

print("Best Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf_best))
print("Best Random Forest Precision:", precision_score(y_test, y_pred_rf_best, average='weighted'))
print("Best Random Forest Recall:", recall_score(y_test, y_pred_rf_best, average='weighted'))
print("Best Random Forest F1-score:", f1_score(y_test, y_pred_rf_best, average='weighted'))
print("Best Random Forest ROC-AUC:", roc_auc_score(y_test, y_pred_proba_rf_best, multi_class='ovr'))

# SHAP Analysis for Random Forest
explainer_rf = shap.Explainer(best_rf, X_train)
shap_values_rf = explainer_rf(X_test)
shap.summary_plot(shap_values_rf, X_test, feature_names=X_test.columns)

# Save Random Forest Model
joblib.dump(best_rf, "best_random_forest_model.pkl")

"""2)XGBoost"""

X_train = pd.get_dummies(X_train, columns=['age_group'])
X_test = pd.get_dummies(X_test, columns=['age_group'])


xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# Hyperparameter Grid
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid Search
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_

# Evaluate Model
y_pred_xgb_best = best_xgb.predict(X_test)
y_pred_proba_xgb_best = best_xgb.predict_proba(X_test)[:, 1]

print("\nBest XGBoost Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb_best))
print("Precision:", precision_score(y_test, y_pred_xgb_best, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_xgb_best, average='weighted'))
print("F1-score:", f1_score(y_test, y_pred_xgb_best, average='weighted'))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba_xgb_best))

# SHAP Analysis
explainer_xgb = shap.Explainer(best_xgb, X_train)
shap_values_xgb = explainer_xgb(X_test)
shap.summary_plot(shap_values_xgb, X_test, feature_names=X_test.columns)

# Save Model
joblib.dump(best_xgb, "best_xgboost_model.pkl")

"""3)BERT LIKE TRANFORMER"""

for feature in features:
    df[feature] = df[feature].astype(str)

# Combine features into a single string for BERT input
df["combined_features"] = df[features].agg(' '.join, axis=1)

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(df["combined_features"], df[target], test_size=0.2, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create DataLoader
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = MentalHealthDataset(
        texts=df.texts.to_numpy(),
        labels=df.labels.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

train_df = pd.DataFrame({'texts': X_train, 'labels': y_train})
test_df = pd.DataFrame({'texts': X_test, 'labels': y_test})

# Define DataLoader parameters
MAX_LEN = 128
BATCH_SIZE = 16

train_data_loader = create_data_loader(train_df, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(test_df, tokenizer, MAX_LEN, BATCH_SIZE)

from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # Import tqdm for progress bar
import torch

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

EPOCHS=3

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training function with progress bar
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training", leave=False):  # Add progress bar
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation function with progress bar
def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating", leave=False):  # Add progress bar
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

best_accuracy = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 10)

    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler)
    print(f"Train loss: {train_loss} | Train accuracy: {train_acc}")

    val_acc, val_loss = eval_model(model, test_data_loader, device)
    print(f"Val loss: {val_loss} | Val accuracy: {val_acc}")

    # Save the model if the validation accuracy is improved
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        model.save_pretrained('best_bert_model')
        tokenizer.save_pretrained('best_bert_model')

    print()

# Save the final model
model.save_pretrained('final_bert_model')
tokenizer.save_pretrained('final_bert_model')

# Model Evaluation
y_pred = []
y_true = []

model = model.eval()
with torch.no_grad():
    for d in test_data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        _, preds = torch.max(outputs.logits, dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("BERT Model Evaluation")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")