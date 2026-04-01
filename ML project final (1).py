import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. LOAD DATASET
# ==========================================
data = pd.read_csv("ML dataset.csv")

# ==========================================
# 2. EDA
# ==========================================
print("First 5 rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

print("\nClass Distribution:")
print(data["Classification"].value_counts())

# Histogram
data["Asteroid Magnitude"].hist()
plt.title("Asteroid Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Count")
plt.show()

# ==========================================
# 3. DATA CLEANING
# ==========================================
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    data[col] = data[col].fillna(data[col].mean())

cat_cols = data.select_dtypes(include=["object"]).columns
for col in cat_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

data.drop_duplicates(inplace=True)

if "Object Name" in data.columns:
    data.drop("Object Name", axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = le.fit_transform(data[col])

# Final check: no NaN should remain
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

# Readable output column
data["Hazardous_Status"] = data["Classification"].map({
    1: "Hazardous",
    0: "Non-Hazardous"
})

print("\nHazardous Status Preview:")
print(data[["Classification", "Hazardous_Status"]].head())

# ==========================================
# 4. INPUT OUTPUT SPLIT
# ==========================================
X = data.drop(["Classification", "Hazardous_Status"], axis=1)
y = data["Classification"]

# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save original test values for final output table
X_test_original = X_test.copy()

# ==========================================
# 6. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 7. LOGISTIC REGRESSION
# ==========================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_acc)
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# ==========================================
# 8. SVM
# ==========================================
svm = SVC()
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("\nSVM Accuracy:", svm_acc)
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# ==========================================
# 9. KNN
# ==========================================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", knn_acc)
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# ==========================================
# 10. BEST MODEL
# ==========================================
accuracies = {
    "Logistic Regression": lr_acc,
    "SVM": svm_acc,
    "KNN": knn_acc
}

best_model = max(accuracies, key=accuracies.get)
print("\nBest Model is:", best_model)

# ==========================================
# 11. FINAL PREDICTION OUTPUT TABLE
# ==========================================
if best_model == "Logistic Regression":
    final_pred = lr_pred
elif best_model == "SVM":
    final_pred = svm_pred
else:
    final_pred = knn_pred

final_output = X_test_original.copy()
final_output["Actual"] = y_test.values
final_output["Predicted"] = final_pred
final_output["Hazard Prediction"] = final_output["Predicted"].map({
    1: "Hazardous",
    0: "Non-Hazardous"
})

print("\nFinal Prediction Output:")
print(final_output.head())