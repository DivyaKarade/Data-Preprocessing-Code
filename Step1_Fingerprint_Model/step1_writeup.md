#Deep Learning for Virtual Screening: WDR91 Binding Prediction with ECFP4

##🔍 Summary Sentence
A deep learning classifier was trained using ECFP4 fingerprints and used to rank 339K molecules for binding to WDR91, selecting top 200 and 500 candidates based on predicted probability scores. 

##📚 Background / Introduction
The challenge involved predicting active small molecules targeting WDR91 using a labelled DEL dataset containing actives and inactives. This high-imbalance dataset (∼30K actives vs 300K inactives) required the design of a robust classifier capable of distinguishing sparse actives based only on chemical fingerprints.
We developed and applied a deep learning model using ECFP4 fingerprints from the DEL training set to classify active vs inactive molecules for WDR91. Our top predictions from the ~339K test set were selected based on model scores to identify 200 and 500 putative binders.
Our goal was to utilise Extended Connectivity Fingerprints (ECFP4) with a neural network classifier to learn the chemical patterns associated with activity, and apply this model to a large test set of ∼339K compounds with the same feature types but no labels.

##🛠 Methods
###⚗️ Data Preprocessing
####Training Set (Step1-TrainingData.parquet)
* Selected the ECFP4 fingerprint type and corresponding LABEL (0: inactive, 1: active).
* Removed outliers using PCA (n=3) + Isolation Forest (contamination=0.4).
* Scaled features using StandardScaler.

####Test Set (Step1_TestData_Target2035.parquet)
* Extracted the ECFP4 fingerprint and RandomID columns.
* Applied the same scaling parameters as the training set.

###🧠 Model Architecture
* Type: Fully Connected Deep Neural Network (Keras Sequential)
* A feed-forward deep neural network (DNN) was trained using Keras with the following configuration:
* Input: ECFP4 fingerprint (size 2048)
* Layers:
1. Dense(500, ReLU) + L2 regularization + Dropout(0.1)
2. Dense(200, ReLU) + L2 regularization + Dropout(0.1)
3. Output: Dense(1, Sigmoid)
* Loss Function: Binary Crossentropy
* Optimizer: Adam
* Loss: Binary Crossentropy
* Callback: EarlyStopping (patience=20, restore_best_weights=True)
* Epochs: Up to 100

###🧪 Training Setup
* Train/Test Split: 75/25
* Outlier detection on both sets using PCA + Isolation Forest
* GPU/CPU compatible via TensorFlow
* Set random seeds (np, tf) for reproducibility

##🧮 Computational Workflow
1. Load Training Data: Read ECFP4 and label columns
2. Preprocess Data: Standardize, remove outliers, split
3. Model Training: Fit deep learning classifier with early stopping
4. Model Evaluation: Accuracy, precision, recall, F1, AUC, confusion matrix
5. Test Data Prediction: Transform and predict probabilities on all ~339K test compounds
6. Ranking: Sort by predicted score
7. Selection:
* Flag top 200 and top 500 predictions as Sel_200 = 1 and Sel_500 = 1 respectively
8. Submission File: Save 4-column .csv with RandomID, Sel_200, Sel_500, Score

##📈 Model Evaluation
The neural network trained on ECFP4 fingerprints showed strong classification performance on the internal validation set, with the following metrics:
* Train Accuracy: 97.80%
* Test Accuracy: 97.47%
* ROC AUC Score: 0.9582
* Precision (Active class): 0.75
* Recall (Sensitivity, Active class): 0.57
* Specificity (Inactive class): 1.00
* F1 Score (Active class): 0.65
* Cohen’s Kappa: 0.64
* Matthews Correlation Coefficient (MCC): 0.64
* G-Mean: 0.75

###Confusion Matrix
|                     | Predicted Inactive | Predicted Active |
| ------------------- | ------------------ | ---------------- |
| **Actual Inactive** | 53,852             | 447              |
| **Actual Active**   | 983                | 1,325            |

###Classification Report
* Inactive: Precision = 0.98, Recall = 1.00, F1-score = 0.99
* Active: Precision = 0.75, Recall = 0.57, F1-score = 0.65
* Macro Avg: Precision = 0.86, Recall = 0.78, F1-score = 0.82
* Weighted Avg: Precision = 0.97, Recall = 0.97, F1-score = 0.97

While the model achieved excellent overall classification metrics, the recall for the active class remained modest due to class imbalance (only ~7% of the training data were actives). Techniques like resampling or class weighting could be explored to enhance minority class performance in future iterations. ROC curves, confusion matrices, and learning curves were visualised.

💾 Output Format
Submission CSV contains:
| Column Name | Description                        |
| ----------- | ---------------------------------- |
| RandomID    | Matches anonymized test IDs        |
| Sel\_200    | Binary (1 if among top 200 scores) |
| Sel\_500    | Binary (1 if among top 500 scores) |
| Score       | Probability output from the model (range [0, 1])  |

##🔄 Resilience and Reproducibility
* Early stopping helps prevent overfitting
* Outlier removal improves signal-to-noise ratio
* Feature scaling ensures model stability
* Code is modular, deterministic (fixed seeds), and exportable

##🔬 Discussion
Our approach relies solely on chemical features (ECFP4) and deep learning to learn activity patterns in a class-imbalanced setup. The model generalised well to unseen data, as reflected by its performance on the hold-out test set.
We avoided oversampling and instead filtered noise via PCA + Isolation Forest. While this may have reduced coverage of actives, it improved signal purity. Given the sparse actives in the test set, our ranking model was tuned to prioritise precision over recall, helping improve the diversity metric in top selections.

##📚Source Code: [Link to Github file](https://github.com/DivyaKarade/First-DREAM-Target-2035-Drug-Discovery-Challenge/blob/main/Step1_Fingerprint_Model/Step1_sourcecode-TesseractalGene.py)

##✍️ Author Statement
Dr. Divya Karade designed and implemented the fingerprint-based deep learning pipeline, performed preprocessing, outlier detection, and generated the final submission file.
