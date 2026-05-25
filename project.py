import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

"""import joblib"""

# Load dataset
df = pd.read_csv("student_data.csv")

print("Dataset:")
print(df)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Features
X = df[["StudyHours", "SleepHours", "Attendance", "PreviousMarks"]]

# Labels
y = df["Pass"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Predict new student
new_student = pd.DataFrame({
    "StudyHours": [6],
    "SleepHours": [7],
    "Attendance": [75],
    "PreviousMarks": [65]
})

# Scale new student data
new_student_scaled = scaler.transform(new_student)

# Prediction
result = model.predict(new_student_scaled)

print("\nPrediction for New Student:", result)

# Save model
"""joblib.dump(model, "performance_model.pkl")

print("\nModel Saved Successfully")"""

# Visualization
plt.scatter(df["StudyHours"], df["PreviousMarks"])

plt.xlabel("Study Hours")
plt.ylabel("Previous Marks")

plt.title("Study Hours vs Previous Marks")

plt.show()

#save model
import joblib

joblib.dump(model, "performance_model.pkl")

print("Model Saved Successfully")