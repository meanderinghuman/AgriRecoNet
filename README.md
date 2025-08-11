AgriRecoNet: Agricultural Recommendation Network
A clean, reproducible benchmark for the classic Crop Recommendation problem using multiple machine learning baselines (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, XGBoost) and a TensorFlow/Keras ANN. Includes EDA, preprocessing, scaling, evaluation via accuracy/F1, confusion matrices, and ROC curves.

✨ Highlights
End-to-end notebook: EDA ➜ preprocessing ➜ train/test ➜ evaluation

Baselines: LR, KNN, SVM, DT, RF, XGBoost

Deep model: ANN (Keras Sequential)

Visuals: correlation heatmap, confusion matrix, ROC

Ready to run locally (requirements included)

📦 Dataset
Name: Crop Recommendation Dataset

Source path used in notebook: Crop_recommendation.csv (or Kaggle path)
Replace the path in the notebook with your local path if needed.

Features typically include soil nutrients (N, P, K), temperature, humidity, pH, rainfall, etc.
Target is the recommended crop label.

Note: The notebook supports either a local CSV (e.g., data/Crop_recommendation.csv) or a Kaggle dataset path. Create a data/ folder and place the CSV there, or update the path in the "Data Load" cell.

🛠️ Environment
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
pip install -r requirements.txt
▶️ Run
Open the notebook in VS Code or Jupyter and run all cells.
If you use VS Code:

Install the Python and Jupyter extensions.

Select your .venv interpreter.

Run cells with Shift+Enter.

🧪 Models Included
Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

XGBoost

Artificial Neural Network (ANN, Keras/TensorFlow)

🧪 Results
Model	Accuracy (%)
Naive Bayes	99.64
Random Forest	99.45
XGBoost	99.45
Gradient Boosting	99.09
Decision Tree	98.73
Artificial Neural Network (ANN)	98.18
SVM	98.00
Logistic Regression	97.27
KNN	95.82

📈 Evaluation
Accuracy, F1 (macro/weighted as configured)

Confusion Matrix

ROC / AUC (one-vs-rest for multi-class)

🗂️ Repo Structure
Copy
Edit
AgriRecoNet/
├─ AgriRecoNet.ipynb
├─ requirements.txt
├─ .gitignore
└─ README.md
🔑 Keywords
agriculture machine learning, crop recommendation, soil nutrients, nitrogen phosphorous potassium, pH, rainfall, temperature, humidity, XGBoost, Random Forest, SVM, KNN, ANN, TensorFlow, Keras, SHAP, EDA, classification

📌 Notes
If you run into GPU errors with TensorFlow, switch to CPU or install the appropriate CUDA/cuDNN combo.

XGBoost is optional; you can comment it out if installation is an issue on your machine.

🧭 Roadmap
Hyperparameter sweeps (Optuna/Scikit-Optimize)

Cross-validation with stratification

Model explainability (SHAP) across all models

Export best model as a saved artifact