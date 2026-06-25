# summative_crop_recommendation_mlp
Table of Contents
Project Overview
Dataset
Methodology
Experiments
Results
Installation & Dependencies
Usage
Repository Structure
Report & Video
License

Project Overview
Goal – Recommend the most suitable crop from 22 classes given soil nutrient levels (N, P, K), temperature, humidity, pH, and rainfall.

Approach – Implement 8 experiments:

Logistic Regression (linear baseline)
Decision Tree (unconstrained)
Random Forest (300 trees)
Gradient Boosting (300 trees, lr=0.05)
Simple MLP (Sequential API, 2 hidden layers)
Deeper MLP with Dropout (Functional API, 3 hidden layers)
MLP + BatchNorm + Early Stopping
Optimized MLP with learning rate scheduling + Early Stopping

Key techniques – Feature engineering (ratio, interaction, composite features), StandardScaler, tf.data pipelines, dropout, batch normalization, ReduceLROnPlateau, and EarlyStopping. All models are evaluated using accuracy, precision, recall, F1-score, ROC‑AUC, confusion matrices, and learning curves.

Dataset
Source: Crop Recommendation Dataset (Kaggle)

Size: 2,200 samples, 7 features, 22 balanced crop classes

Features: N, P, K, temperature, humidity, pH, rainfall

Splits: 70% training, 15% validation, 15% test (stratified)

Methodology
Preprocessing: No missing values; features standardized using StandardScaler (fit only on training data).

Feature Engineering: Created 9 additional features (N:P ratio, N:K ratio, P:K ratio, NPK sum, NPK product, N×pH, P×pH, K×pH, pH²) based on agronomic principles.

Data Pipeline: Used tf.data with shuffling, batching (32), and prefetching for efficient training.

Evaluation: Macro‑averaged metrics, confusion matrices, ROC curves, and learning curves.

Experiments
Exp	Model	Key Hyperparameters / Architecture
1	Logistic Regression	C=1.0, max_iter=1000, lbfgs
2	Decision Tree	max_depth=None
3	Random Forest	n_estimators=300, max_depth=None
4	Gradient Boosting	n_estimators=300, lr=0.05, max_depth=5, subsample=0.8
5	Simple MLP (Sequential)	2 hidden layers [256, 128], ReLU, Adam
6	Deeper MLP + Dropout (Functional)	3 hidden layers [256, 128, 64], Dropout(0.3)
7	MLP + BatchNorm + EarlyStop	[256, 128], BatchNorm, Dropout(0.3), EarlyStopping
8	Optimized MLP (LR Schedule + EarlyStop)	[256, 128, 64], BatchNorm, Dropout(0.3/0.2), ReduceLROnPlateau, EarlyStopping
All neural networks use the Adam optimizer, sparse_categorical_crossentropy loss, and tf.data pipelines.

Results
Experiment	Accuracy	Precision	Recall	F1‑Score
Logistic Regression	0.9515	0.9515	0.9515	0.9515
Decision Tree	0.9788	0.9788	0.9788	0.9788
Random Forest	0.9909	0.9909	0.9909	0.9909
Gradient Boosting	0.9939	0.9939	0.9939	0.9939
Simple MLP	0.9727	0.9727	0.9727	0.9727
Deeper MLP + Dropout	0.9848	0.9848	0.9848	0.9848
MLP + BatchNorm	0.9909	0.9909	0.9909	0.9909
Optimized MLP	0.9970	0.9970	0.9970	0.9970
The Optimized MLP (Exp 8) achieved the best performance with only one misclassification out of 330 test samples.

Gradient Boosting and Random Forest are excellent alternatives with high accuracy and interpretability.

Feature engineering (ratio and interaction features) significantly boosted all models, especially logistic regression.

Installation & Dependencies
Prerequisites
Python 3.8+

Jupyter Notebook / Google Colab

Install required packages
bash
pip install -r requirements.txt
requirements.txt
text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
kagglehub

Usage
Clone the repository:

bash
git clone https://github.com/Mukasa-E/summative_crop_recommendation_mlP.git
cd summative_crop_recommendation_mlP
Open the Jupyter notebook:

bash
jupyter notebook Crop_Recommendation_Experiments.ipynb
Run all cells from top to bottom. The notebook will automatically download the dataset using kagglehub, perform preprocessing, run all 8 experiments, and generate all visualizations (confusion matrices, learning curves, ROC curves, feature importance, comparison charts).

Results are printed within the notebook; images are saved as confusion_matrices.png, learning_curves.png, and results_comparison.png.

All random seeds are fixed (SEED = 42) for full reproducibility.


Acknowledgments
Dataset provided by Atharva Ingle on Kaggle.

Built with TensorFlow, Scikit‑learn, and other open‑source libraries.

Author: [Emmanuel Mukasa]
Date: 23,June 2026
Course: Introduction to Machine Learning (Summative Project)
