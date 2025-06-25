# --- Imports ---
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, precision_score,
    recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, accuracy_score
)
from sklearn.metrics import classification_report
from read_parquet_utils import read_parquet_file, process_column_to_array
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Reproducibility ---
np.random.seed(1)
tf.random.set_seed(3)

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Session configuration for TensorFlow 2.x
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

# Force TensorFlow to use a single thread
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

# --- Load Data ---
train_df = read_parquet_file('Step1-TrainingData.parquet', columns=['ECFP4', 'LABEL'])
# Display basic info about the data
print("Columns:", train_df.columns)
print("Data shape:", train_df.shape)
print(train_df.head())
test_df = read_parquet_file('Step1_TestData_Target2035.parquet', columns=['RandomID', 'ECFP4'])
# Display basic info about the data
print("Columns:", test_df.columns)
print("Data shape:", test_df.shape)
print(test_df.head())

# Convert fingerprints to NumPy arrays
X = process_column_to_array(train_df, 'ECFP4')
# Labels are already integers (0/1)
y = train_df['LABEL'].values
# Convert fingerprints to NumPy arrays
X_test_all = process_column_to_array(test_df, 'ECFP4')

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)

# --- Scaling ---
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --- Outlier Removal ---
iso = IsolationForest(contamination=0.4, random_state=3)
pca = PCA(n_components=3)

# Train set filtering
pca_train = pca.fit_transform(X_train)
mask_train = iso.fit_predict(pca_train) != -1
X_train, y_train = X_train[mask_train], y_train[mask_train]

# Test set filtering
pca_test = pca.fit_transform(X_test)
mask_test = iso.fit_predict(pca_test) != -1
X_test, y_test = X_test[mask_test], y_test[mask_test]

print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# --- Model Definition ---
model = Sequential([
    Dense(500, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.1),
    Dense(200, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Training ---
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_data=(X_test, y_test),
                    callbacks=[early_stop], verbose=1)

# --- Evaluation ---
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Train Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# --- Predictions ---
probs = model.predict(X_test).flatten()
preds = (probs > 0.5).astype(int)

# Plot Learning Curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# plot accuracy:
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, preds)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Inactive", "Active"]))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Inactive', 'Active'], yticklabels=['Inactive', 'Active'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, probs)
auc_score = roc_auc_score(y_test, probs)

print(f"ROC AUC Score: {auc_score:.4f}")

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
print("AUC:", roc_auc_score(y_test, probs))

# --- Metrics ---
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)
kappa = cohen_kappa_score(y_test, preds)
mcc = matthews_corrcoef(y_test, preds)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
gmeans = math.sqrt(specificity * recall)

print("Classification Report:")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Cohen's Kappa: {kappa:.2f}")
print(f"Matthews Correlation Coefficient: {mcc:.2f}")
print(f"G-Mean: {gmeans:.2f}")

# --- Final Predictions ---
X_final_test = sc.transform(X_test_all)
final_probs = model.predict(X_final_test).flatten()
test_df['Score'] = final_probs
test_df = test_df.sort_values('Score', ascending=False).reset_index(drop=True)
test_df['Sel_200'] = 0
test_df['Sel_500'] = 0
test_df.loc[:199, 'Sel_200'] = 1
test_df.loc[:499, 'Sel_500'] = 1

# --- Save Submission ---
TEAM_NAME = 'TeamTesseractalGene'
submission = test_df[['RandomID', 'Sel_200', 'Sel_500', 'Score']]
submission.to_csv(f'Team{TEAM_NAME}.csv', index=False)
print(f"Submission saved to Team{TEAM_NAME}.csv")

# --- Model Summary ---
model.summary()
