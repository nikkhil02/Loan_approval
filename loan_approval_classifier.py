# Loan Approval Classifier - VS Code Version
# Converted from the original Jupyter Notebook.
# Run this file from the folder containing loan_data.csv.

# =========================
# 1. IMPORT LIBRARIES
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# 2. LOAD THE DATASET
# =========================
# Loading and displaying the dataset as a Pandas DataFrame
loan_df = pd.read_csv("loan_data.csv")
print("\nFirst 5 rows of the dataset:")
print(loan_df.head())


# =========================
# 3. DATASET INFORMATION
# =========================
# Display the number of rows and columns
print("\nNumber of rows and columns:", loan_df.shape)

# Display summary statistics for numerical variables
print("\nSummary statistics:")
print(loan_df.describe())


# =========================
# 4. CHECK MISSING VALUES
# =========================
print("\nMissing values in each column:")
print(loan_df.isnull().sum())


# =========================
# 5. TARGET VARIABLE DISTRIBUTION
# =========================
sns.countplot(
    x="not.fully.paid",
    data=loan_df,
    hue="not.fully.paid",
    palette=["#0000FF", "#FF5349"],
    legend=False
)
plt.title("Distribution of Target Variable")
plt.show()


# =========================
# 6. CORRELATION HEATMAP
# =========================
# Select only numeric columns
numeric_loan_df = loan_df.select_dtypes(include=["float64", "int64"])

# Calculate the correlation matrix
corr = numeric_loan_df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    annot_kws={"size": 10}
)
plt.title("Correlation Between Variables")
plt.show()


# =========================
# 7. LOAN PURPOSE DISTRIBUTION
# =========================
sns.countplot(
    x="purpose",
    data=loan_df,
    hue="purpose",
    palette="tab10",
    legend=False
)
plt.title("Loan Purpose Distribution")
plt.xticks(rotation=90)
plt.show()


# =========================
# 8. INTEREST RATE BY LOAN PURPOSE
# =========================
sns.boxplot(
    x="purpose",
    y="int.rate",
    data=loan_df,
    hue="purpose",
    palette="tab10",
    legend=False
)
plt.title("Interest Rate by Loan Purpose")
plt.xticks(rotation=90)
plt.show()


# =========================
# 9. FEATURE ENGINEERING
# =========================
# Create installment-to-income ratio
loan_df["installment_to_income_ratio"] = (
    loan_df["installment"] / loan_df["log.annual.inc"]
)

# Create credit history feature
loan_df["credit_history"] = (
    loan_df["delinq.2yrs"] + loan_df["pub.rec"]
) / loan_df["fico"]


# =========================
# 10. PREPROCESSING
# =========================
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop unnecessary columns
loan_df = loan_df.drop(
    ["credit.policy", "days.with.cr.line", "purpose"],
    axis=1
)

# Convert the target variable into numerical values
le = LabelEncoder()
loan_df["not.fully.paid"] = le.fit_transform(loan_df["not.fully.paid"])


# =========================
# 11. SCALE NUMERICAL FEATURES
# =========================
scaler = StandardScaler()

numerical_cols = [
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
    "credit_history",
    "installment_to_income_ratio"
]

loan_df[numerical_cols] = scaler.fit_transform(loan_df[numerical_cols])


# =========================
# 12. HANDLE CLASS IMBALANCE
# =========================
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)

X = loan_df.drop("not.fully.paid", axis=1)
y = loan_df["not.fully.paid"]

X_resampled, y_resampled = sm.fit_resample(X, y)

loan_df = pd.concat([X_resampled, y_resampled], axis=1)

print("\nClass distribution after SMOTE:")
print(loan_df["not.fully.paid"].value_counts())


# =========================
# 13. TRAIN-TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X = loan_df.drop("not.fully.paid", axis=1)
y = loan_df["not.fully.paid"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# =========================
# 14. MODEL SELECTION
# =========================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)
print("\nDecision Tree Classifier Accuracy: {:.2f}%".format(dt_score * 100))

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)
print("Random Forest Classifier Accuracy: {:.2f}%".format(rf_score * 100))

# Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)
print("Logistic Regression Classifier Accuracy: {:.2f}%".format(lr_score * 100))

# Support Vector Machine
svm = SVC(random_state=42)
svm.fit(X_train, y_train)
svm_score = svm.score(X_test, y_test)
print("Support Vector Machine Classifier Accuracy: {:.2f}%".format(svm_score * 100))


# =========================
# 15. HYPERPARAMETER TUNING
# =========================
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Parameter grid for Random Forest
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="f1"
)

grid_search.fit(X_train, y_train)

# Get the best model and hyperparameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_


# =========================
# 16. MODEL EVALUATION
# =========================
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nRandom Forest Classifier Evaluation Results:")
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))
print("Best hyperparameters:", best_params)


# =========================
# 17. SAVE THE BEST MODEL
# =========================
import joblib

joblib.dump(best_model, "loan_classifier.joblib")
print("\nBest model saved as loan_classifier.joblib")

