
# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# load processed csv from day 1 and 2 
df_model = pd.read_csv("clean_sample.csv")

# seperate targets
y = df_model["Label"]

# numeric features only for x
X = df_model.select_dtypes(include=np.number)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# predictions
y_pred = rf.predict(X_test)

# evaluation results 
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='binary', pos_label=y.unique()[1])
rec = recall_score(y_test, y_pred, average='binary', pos_label=y.unique()[1])
f1 = f1_score(y_test, y_pred, average='binary', pos_label=y.unique()[1])

print("\nRandom Forest Evaluation")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# cross-validate metrics
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("\nCross-Validation Results")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"CV Standard Deviation: {cv_scores.std():.4f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=y.unique(), yticklabels=y.unique())
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# top 10 feature importance 
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(8, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

import joblib

# Save trained model and feature names
joblib.dump(rf, "rf_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("\nModel and feature names saved successfully!")
