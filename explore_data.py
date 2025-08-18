# import statements 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading dataset, note use of "r" to avoid unicode errors 
csv_file = r"C:\Users\pkrao\ML_IDS_PROJECT_NETRA\TrafficLabelling\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
df = pd.read_csv(csv_file, low_memory=False)

# Column names & strip
df.columns = df.columns.str.strip()

# Identify label's column
label_col = None
for c in df.columns:
    if c.strip().lower() == "label":
        label_col = c
        break
if not label_col:
    raise ValueError("Couldn't find a 'Label' column. Check CSV headers.")

print("Dataset loaded from:", csv_file)
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# Label distribution
print("\n=== Label distribution ===")
print(df[label_col].value_counts())

# Check missing values 
print("\nMissing values per column (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

# Replace infinity string NaN with np.nan
df.replace([np.inf, -np.inf, "Infinity", "NaN", "nan", "None", ""], np.nan, inplace=True)

# Convert object columns to numeric 
skip_objects = {label_col, "Flow ID", "Source IP", "Destination IP", "Src IP", "Dst IP", "Timestamp"}
for col in df.columns:
    if df[col].dtype == "object" and col not in skip_objects:
        df[col] = pd.to_numeric(df[col], errors="ignore")

# Fill numeric NaNs with median
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Droping rows with missing labels or duplicates
df = df.dropna(subset=[label_col])
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Dropped duplicates: {before - len(df)}")

print("\nMissing values after cleaning (nonzero only):")
nz = df.isna().sum()
print(nz[nz > 0])

#Categoric labels for modelling : one hot encoding 

obj_cols = df.select_dtypes(include="object").columns
exclude = {label_col, "Flow ID", "Source IP", "Destination IP", "Src IP", "Dst IP", "Timestamp"}
encode_cols = [c for c in obj_cols if c not in exclude]
df_model = pd.get_dummies(df, columns=encode_cols, drop_first=True)
print("\nShape after one-hot encoding:", df_model.shape)

# Binary target attack vs benign
df["target_binary"] = (df[label_col].astype(str).str.upper() != "BENIGN").astype(int)
print("\nBinary target counts (1=attack, 0=benign):")
print(df["target_binary"].value_counts())

# Visualing label distributing 
plt.figure(figsize=(10,6))
sns.countplot(x=label_col, data=df)
plt.xticks(rotation=45, ha="right")
plt.title("Distribution of Labels")
plt.tight_layout()
plt.show()

# Clean up 
sample = df.sample(min(50000, len(df)), random_state=42)
sample.to_csv("clean_sample.csv", index=False)
print("\nSaved cleaned sample to clean_sample.csv")




#Day 2 code: 

print("\n Model training  (using df_model numeric features only)")

#Make sure target exists
if "target_binary" not in df.columns:
    df["target_binary"] = (df[label_col].astype(str).str.upper() != "BENIGN").astype(int)

# Build x from df model 
# Keep numeric columons only, drop target 
X = df_model.select_dtypes(include=[np.number]).copy()
X = X.drop(columns=["target_binary"], errors="ignore")
y = df["target_binary"].astype(int).loc[X.index]

# Clean up inf/NaN
X = X.replace([np.inf, -np.inf], np.nan)
if X.isna().any().any():
    X = X.fillna(X.median(numeric_only=True))

print(f"Number of numeric features used: {X.shape[1]}")

#Split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

#Scale so training models can recognize data 
scaler = StandardScaler(with_mean=True, with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Models
models = {"Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),"Decision Tree": DecisionTreeClassifier(random_state=42),"Logistic Regression": LogisticRegression(solver="saga", max_iter=200, n_jobs=-1, random_state=42),"Linear SVM": LinearSVC(random_state=42, max_iter=5000)}

#Evaluate model performance 
for name, model in models.items():
    print(f"\n_________{name}_________")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["BENIGN(0)", "ATTACK(1)"]))



#Decision: Random Forest Training model: achieved 99.99% accuracy, handled 80 numeric features complex attack-benign features, and resists overfitting 