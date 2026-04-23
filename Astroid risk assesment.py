import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from xgboost import XGBClassifier

# ==========================================
# SETTINGS
# ==========================================
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
RANDOM_STATE = 42

# ==========================================
# 1. LOAD DATASET
# ==========================================
data = pd.read_csv("ML dataset.csv")

print("First 5 Rows:")
print(data.head())

print("\nDataset Shape Before Cleaning:", data.shape)
print("\nMissing Values Before Cleaning:")
print(data.isnull().sum())

# ==========================================
# 2. DATA CLEANING
# ==========================================
num_cols_all = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols_all = data.select_dtypes(include=["object"]).columns.tolist()

for col in num_cols_all:
    data[col] = data[col].fillna(data[col].mean())

for col in cat_cols_all:
    data[col] = data[col].fillna(data[col].mode()[0])

data.drop_duplicates(inplace=True)

if "Object Name" in data.columns:
    data.drop("Object Name", axis=1, inplace=True)

for col in data.select_dtypes(include=["object"]).columns:
    if col != "Classification":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

print("\nDataset Shape After Cleaning:", data.shape)

# ==========================================
# 3. CREATE HAZARDOUS TARGET
# Hazardous = 1 if MOID <= 0.05 and Magnitude <= 22
# ==========================================
moid_candidates = [
    "Minimum Orbit Intersection Distance (AU)",
    "Minimum Orbit Intersection Distance",
    "MOID (AU)",
    "MOID"
]

mag_candidates = [
    "Asteroid Magnitude",
    "Absolute Magnitude",
    "Magnitude"
]

moid_col = None
mag_col = None

for col in moid_candidates:
    if col in data.columns:
        moid_col = col
        break

for col in mag_candidates:
    if col in data.columns:
        mag_col = col
        break

if moid_col is None or mag_col is None:
    raise ValueError(
        "Required columns not found for Hazardous creation. "
        "Please check MOID and Magnitude column names."
    )

data["Hazardous"] = np.where(
    (data[moid_col] <= 0.05) & (data[mag_col] <= 22),
    1,
    0
)

print("\nHazardous Target Distribution:")
print(data["Hazardous"].value_counts())

# ==========================================
# 4. EDA GRAPHS
# ==========================================

# Graph 1
plt.figure(figsize=(8, 5))
sns.countplot(x="Hazardous", data=data)
plt.title("Hazardous vs Non-Hazardous Asteroids")
plt.xlabel("0 = Non-Hazardous | 1 = Hazardous")
plt.ylabel("Count")
plt.show()

# Graph 2
plt.figure(figsize=(8, 5))
sns.histplot(data[mag_col], bins=30, kde=True)
plt.title("Asteroid Magnitude Distribution")
plt.xlabel(mag_col)
plt.ylabel("Count")
plt.show()

# Graph 4
plt.figure(figsize=(8, 5))
sns.boxplot(x="Hazardous", y=mag_col, data=data)
plt.title("Magnitude vs Hazardous Label")
plt.xlabel("Hazardous")
plt.ylabel(mag_col)
plt.show()

# Graph 5
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data.sample(min(1200, len(data)), random_state=RANDOM_STATE),
    x=moid_col,
    y=mag_col,
    hue="Hazardous",
    palette="coolwarm",
    alpha=0.7
)
plt.title("MOID vs Magnitude")
plt.show()

# Graph 6
plt.figure(figsize=(12, 8))
sns.heatmap(
    data.select_dtypes(include=np.number).corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Feature Correlation Heatmap")
plt.show()

# ==========================================
# 5. INPUT / OUTPUT SPLIT
# Remove Hazardous, Classification, MOID and Magnitude from X
# to avoid leakage
# ==========================================
drop_cols = ["Hazardous"]
if "Classification" in data.columns:
    drop_cols.append("Classification")
drop_cols.extend([moid_col, mag_col])

X = data.drop(columns=drop_cols, errors="ignore")
y = data["Hazardous"]

# Only for decision boundary graphs
X_vis = data[[moid_col, mag_col]].copy()

print("\nFeatures used for training:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

# ==========================================
# 6. TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

Xvis_train, Xvis_test, yvis_train, yvis_test = train_test_split(
    X_vis,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining Size:", len(X_train))
print("Testing Size:", len(X_test))
print("Total Dataset Size:", len(data))

# ==========================================
# 7. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_full_scaled = scaler.transform(X)

vis_scaler = StandardScaler()
Xvis_train_scaled = vis_scaler.fit_transform(Xvis_train)
Xvis_test_scaled = vis_scaler.transform(Xvis_test)
Xvis_full_scaled = vis_scaler.transform(X_vis)

# ==========================================
# 8. COMMON FUNCTIONS
# ==========================================
results = []

def evaluate_model(y_true, y_pred, y_prob, model_name):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n===================================")
    print(model_name)
    print("===================================")
    print("Accuracy  :", round(acc * 100, 2), "%")
    print("Precision :", round(pre * 100, 2), "%")
    print("Recall    :", round(rec * 100, 2), "%")
    print("F1 Score  :", round(f1 * 100, 2), "%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": roc_auc
    })

    return acc, pre, rec, f1, roc_auc

def plot_full_confusion_matrix(y_full, pred_full, title):
    cm = confusion_matrix(y_full, pred_full)

    print("\n", title)
    print(cm)
    print("Sum of confusion matrix =", cm.sum())
    print("Dataset size =", len(y_full))

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=1,
        linecolor="black",
        xticklabels=["Non-Hazardous", "Hazardous"],
        yticklabels=["Non-Hazardous", "Hazardous"]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def plot_decision_boundary(model, X_plot, y_plot, title, x_label, y_label):
    model.fit(X_plot, y_plot)

    x1 = X_plot[:, 0]
    x2 = X_plot[:, 1]

    x1_min, x1_max = np.percentile(x1, 1), np.percentile(x1, 99)
    x2_min, x2_max = np.percentile(x2, 1), np.percentile(x2, 99)

    xx, yy = np.meshgrid(
        np.linspace(x1_min - 0.5, x1_max + 0.5, 300),
        np.linspace(x2_min - 0.5, x2_max + 0.5, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
    plt.contour(xx, yy, Z, colors="blue", linewidths=1)

    rng = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_plot), size=min(600, len(X_plot)), replace=False)

    if hasattr(y_plot, "iloc"):
        y_scatter = y_plot.iloc[sample_idx]
    else:
        y_scatter = y_plot[sample_idx]

    plt.scatter(
        X_plot[sample_idx, 0],
        X_plot[sample_idx, 1],
        c=y_scatter,
        cmap="coolwarm",
        edgecolors="black",
        s=35
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

# ==========================================
# 9. HANDLE CLASS IMBALANCE
# ==========================================
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight_value = neg_count / pos_count

print("\nClass imbalance info:")
print("Non-Hazardous:", neg_count)
print("Hazardous:", pos_count)
print("scale_pos_weight:", round(scale_pos_weight_value, 3))

# ==========================================
# 10. TRAIN MODELS
# ==========================================

# Logistic Regression
lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
lr_acc, lr_pre, lr_rec, lr_f1, lr_auc = evaluate_model(
    y_test, lr_pred, lr_prob, "Logistic Regression"
)

# SVM
svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_prob = svm.predict_proba(X_test_scaled)[:, 1]
svm_acc, svm_pre, svm_rec, svm_f1, svm_auc = evaluate_model(
    y_test, svm_pred, svm_prob, "SVM"
)

# KNN
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(X_train_scaled, y_train)
knn_pred = knn.predict(X_test_scaled)
knn_prob = knn.predict_proba(X_test_scaled)[:, 1]
knn_acc, knn_pre, knn_rec, knn_f1, knn_auc = evaluate_model(
    y_test, knn_pred, knn_prob, "KNN"
)

# Decision Tree
dt = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=40,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)
dt_prob = dt.predict_proba(X_test_scaled)[:, 1]
dt_acc, dt_pre, dt_rec, dt_f1, dt_auc = evaluate_model(
    y_test, dt_pred, dt_prob, "Decision Tree"
)

# XGBoost
xgb = XGBClassifier(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2,
    scale_pos_weight=scale_pos_weight_value,
    random_state=RANDOM_STATE,
    eval_metric="logloss"
)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)
xgb_prob = xgb.predict_proba(X_test_scaled)[:, 1]
xgb_acc, xgb_pre, xgb_rec, xgb_f1, xgb_auc = evaluate_model(
    y_test, xgb_pred, xgb_prob, "XGBoost"
)

# ==========================================
# 11. MODEL COMPARISON
# BEST MODEL BY F1 SCORE, NOT ACCURACY
# ==========================================
results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "SVM", "KNN", "Decision Tree", "XGBoost"],
    "Accuracy": [lr_acc, svm_acc, knn_acc, dt_acc, xgb_acc],
    "Precision": [lr_pre, svm_pre, knn_pre, dt_pre, xgb_pre],
    "Recall": [lr_rec, svm_rec, knn_rec, dt_rec, xgb_rec],
    "F1 Score": [lr_f1, svm_f1, knn_f1, dt_f1, xgb_f1],
    "AUC": [lr_auc, svm_auc, knn_auc, dt_auc, xgb_auc]
})

print("\nModel Comparison:")
print(results_df)

best_model_name = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]
best_f1 = results_df["F1 Score"].max()
best_accuracy = results_df.loc[results_df["F1 Score"].idxmax(), "Accuracy"]

print("\nWhy accuracy alone is misleading:")
print("If a model predicts mostly Non-Hazardous, accuracy can still be high.")
print("So for this project, F1 Score is used to select the best model.")

print("\nBEST MODEL :", best_model_name)
print("BEST F1 SCORE :", round(best_f1 * 100, 2), "%")
print("ACCURACY OF BEST MODEL :", round(best_accuracy * 100, 2), "%")

# ==========================================
# F1 SCORE COMPARISON GRAPH (IMPORTANT)
# ==========================================

plt.figure(figsize=(10, 6))

sns.barplot(
    x=results_df["Model"],
    y=results_df["F1 Score"],
    palette="viridis"
)

plt.title("Model Comparison Based on F1 Score")
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)

# show values on bars
lr_full_pred = lr.predict(X_full_scaled)

cm_lr = confusion_matrix(y, lr_full_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    xticklabels=["Non-Hazardous", "Hazardous"],
    yticklabels=["Non-Hazardous", "Hazardous"]
)

plt.title("Logistic Regression Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Total =", cm_lr.sum())

#svm confusion matrix
svm_full_pred = svm.predict(X_full_scaled)

cm_svm = confusion_matrix(y, svm_full_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="YlGn",
    xticklabels=["Non-Hazardous", "Hazardous"],
    yticklabels=["Non-Hazardous", "Hazardous"]
)

plt.title("SVM Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Total =", cm_svm.sum())

#knn confusion matrix
knn_full_pred = knn.predict(X_full_scaled)

cm_knn = confusion_matrix(y, knn_full_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_knn,
    annot=True,
    fmt="d",
    cmap="PuBu",
    xticklabels=["Non-Hazardous", "Hazardous"],
    yticklabels=["Non-Hazardous", "Hazardous"]
)

plt.title("KNN Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Total =", cm_knn.sum())

#decision tree confusio matrix
dt_full_pred = dt.predict(X_full_scaled)

cm_dt = confusion_matrix(y, dt_full_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=["Non-Hazardous", "Hazardous"],
    yticklabels=["Non-Hazardous", "Hazardous"]
)

plt.title("Decision Tree Confusion Matrix (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Total =", cm_dt.sum())

# ==========================================
# 12. ONE CONFUSION MATRIX ON FULL DATASET
# Sum will equal full dataset size
# ==========================================
if best_model_name == "Logistic Regression":
    best_model = lr
elif best_model_name == "SVM":
    best_model = svm
elif best_model_name == "KNN":
    best_model = knn
elif best_model_name == "Decision Tree":
    best_model = dt
else:
    best_model = xgb

full_pred = best_model.predict(X_full_scaled)
plot_full_confusion_matrix(y, full_pred, f"{best_model_name} Full Dataset Confusion Matrix")

# ==========================================
# 13. ROC CURVE COMPARISON
# ==========================================
plt.figure(figsize=(8, 6))

for name, probs in [
    ("Logistic Regression", lr_prob),
    ("SVM", svm_prob),
    ("KNN", knn_prob),
    ("Decision Tree", dt_prob),
    ("XGBoost", xgb_prob)
]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {model_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="black")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 14. MODEL COMPARISON BAR GRAPH
# ==========================================
plt.figure(figsize=(11, 6))

metrics_plot = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1 Score", "AUC"],
    var_name="Metric",
    value_name="Score"
)

sns.barplot(data=metrics_plot, x="Model", y="Score", hue="Metric")

plt.title("Model Comparison on Accuracy, Precision, Recall, F1 and AUC")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.legend(title="Metric")
plt.show()

# ==========================================
# 15. MODEL COMPARISON LINE GRAPH
# ==========================================
plt.figure(figsize=(11, 6))

plt.plot(results_df["Model"], results_df["Accuracy"], marker="o", linewidth=2, label="Accuracy")
plt.plot(results_df["Model"], results_df["Precision"], marker="o", linewidth=2, label="Precision")
plt.plot(results_df["Model"], results_df["Recall"], marker="o", linewidth=2, label="Recall")
plt.plot(results_df["Model"], results_df["F1 Score"], marker="o", linewidth=2, label="F1 Score")
plt.plot(results_df["Model"], results_df["AUC"], marker="o", linewidth=2, label="AUC")

plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.show()

# ==========================================
# 16. DECISION BOUNDARY GRAPHS
# Use only MOID and Magnitude for visualization
# ==========================================

plot_decision_boundary(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    Xvis_train_scaled,
    yvis_train,
    "Logistic Regression Decision Boundary",
    "Scaled MOID",
    "Scaled Magnitude"
)

plot_decision_boundary(
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    Xvis_train_scaled,
    yvis_train,
    "SVM Decision Boundary",
    "Scaled MOID",
    "Scaled Magnitude"
)

plot_decision_boundary(
    KNeighborsClassifier(n_neighbors=11),
    Xvis_train_scaled,
    yvis_train,
    "KNN Decision Boundary",
    "Scaled MOID",
    "Scaled Magnitude"
)

plot_decision_boundary(
    XGBClassifier(
        n_estimators=60,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_value,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    ),
    Xvis_train_scaled,
    yvis_train,
    "XGBoost Decision Boundary",
    "Scaled MOID",
    "Scaled Magnitude"
)

# ==========================================
# 17. DECISION TREE VISUALIZATION
# ==========================================
dt_vis = DecisionTreeClassifier(
    criterion="gini",
    max_depth=2,
    min_samples_split=50,
    min_samples_leaf=25,
    class_weight="balanced",
    random_state=RANDOM_STATE
)
dt_vis.fit(Xvis_train_scaled, yvis_train)

plt.figure(figsize=(14, 8))
plot_tree(
    dt_vis,
    filled=True,
    rounded=True,
    feature_names=["Scaled MOID", "Scaled Magnitude"],
    class_names=["Non-Hazardous", "Hazardous"],
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()

# ==========================================
# 18. FINAL OUTPUT COLUMN ADDED
# ==========================================
data["Predicted_Class"] = full_pred
data["Predicted_Label"] = data["Predicted_Class"].map({
    0: "Non-Hazardous",
    1: "Hazardous"
})

print("\nFinal Dataset Preview:")
print(data.head())

data.to_csv("Final_Asteroid_Prediction_Output.csv", index=False)
print("\nFile Saved: Final_Asteroid_Prediction_Output.csv")