import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import auc

# Example: Let's assume you have AI predictions (y_pred) and true values (y_true)
y_true = ['INFP', 'ENFP', 'INTJ', 'INTP', 'INFP', 'ENFP', 'INTJ', 'ENTP', 'INFJ', 'INFP']
y_pred = ['INFP', 'ENFP', 'INFP', 'INTP', 'INFP', 'ENFP', 'INTJ', 'ENFP', 'INFJ', 'INFP']

# Define labels
labels = ['INFP', 'ENFP', 'INTJ', 'INTP', 'INFJ', 'ENTP']

# Binarize the true labels and predicted labels for AUC calculation
y_true_bin = label_binarize(y_true, classes=labels)
y_pred_bin = label_binarize(y_pred, classes=labels)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# --- 1. Confusion Matrix Plot ---
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', cbar=False, xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Personality Type")
plt.ylabel("True Personality Type")
plt.show(block=True)

# --- 2. Bar plot for Accuracy, Precision, Recall ---
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall}
names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(8,6))
sns.barplot(x=names, y=values, palette='viridis')
plt.title('AI Model Evaluation Metrics')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.show(block=True)

# --- 3. ROC AUC Plot ---
# Calculate the ROC curve and AUC for each class
fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show(block=True)

# --- 4. Distribution of predictions (Example) ---
prediction_counts = [3, 2, 1, 2, 1, 1]  # Example data

plt.figure(figsize=(8,6))
sns.barplot(x=labels, y=prediction_counts, palette='coolwarm')
plt.title('Distribution of AI Predictions')
plt.xlabel('Personality Type')
plt.ylabel('Count of Predictions')
plt.show(block=True)

# --- 5. Random Scatter Plot Example ---
x = np.random.rand(100)
y = np.random.rand(100)

plt.figure(figsize=(8,6))
plt.scatter(x, y, color='purple', alpha=0.5)
plt.title('Random Scatter Plot for Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show(block=True)
