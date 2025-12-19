from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# 1. Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create model
model = RandomForestClassifier(random_state=42)

# 4. Train model
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

#  Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Iris Classification")
plt.show()

print("\nConfusion Matrix:")
print(cm)

# 🔹 Classification Report

report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:")
print(report)

# 6. Save model to disk
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("Model saved to model/model.pkl")

