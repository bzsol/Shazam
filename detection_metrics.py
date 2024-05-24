from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# Assuming y_true and y_pred are your true labels and predictions respectively
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Example true labels
y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]  # Example predictions

# Calculate precision, recall, F1-score, and accuracy
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Calculate confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
specificity = tn / (tn + fp)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'Accuracy: {accuracy:.2f}')
print(f'Specificity: {specificity:.2f}')
