# AgriRecoNet: Agricultural Recommendation Network 🌾

A clean, reproducible benchmark for the classic **Crop Recommendation** problem. This project implements and evaluates multiple machine learning baselines and a deep learning model to predict the optimal crop based on environmental and soil features.

![GitHub](https://img.shields.io/github/license/google/gemini-python?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Frameworks](https://img.shields.io/badge/TensorFlow%20%7C%20Scikit--Learn-orange?style=for-the-badge&logo=tensorflow)

---

## ✨ Highlights

This repository provides an end-to-end, ready-to-run solution for crop recommendation.

-   **Comprehensive EDA:** Includes visualizations like a correlation heatmap to understand feature relationships.
-   **Multiple Baselines:** Compares classic ML models: **Logistic Regression, KNN, SVM, Decision Tree, Random Forest, and XGBoost**.
-   **Deep Learning Model:** Features a custom **Artificial Neural Network (ANN)** built with TensorFlow/Keras.
-   **Thorough Evaluation:** Assesses models using accuracy, F1-score, confusion matrices, and ROC/AUC curves.
-   **Reproducible Environment:** Comes with a `requirements.txt` file for easy setup.

---

## 📦 Dataset

The model is trained on the **Crop Recommendation Dataset**, which contains the following features:

-   **Soil Nutrients:** Nitrogen (N), Phosphorous (P), Potassium (K)
-   **Soil Condition:** pH value
-   **Environmental Factors:** Temperature, Humidity, Rainfall

The target variable is the recommended crop label (e.g., Rice, Maize, Chickpea, etc.).

**To Use:**
The notebook is configured to load `Crop_recommendation.csv` from a `data/` directory.

1.  Create a folder named `data` in the root of the project.
2.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) and place it inside the `data/` folder.
3.  Alternatively, update the path in the "Data Load" cell of the notebook to your specific file location.

---

## 🛠️ Setup & Run

Get up and running in a few simple steps.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/AgriRecoNet.git](https://github.com/meanderinghuman/AgriRecoNet.git)
    cd AgriRecoNet
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate it (Linux/Mac)
    source .venv/bin/activate

    # Activate it (Windows)
    .\.venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the notebook:**
    Open the `AgriRecoNet.ipynb` notebook in VS Code or Jupyter and run the cells.
    * **In VS Code:** Ensure you have the Python and Jupyter extensions installed. Select the `.venv` interpreter from the command palette (`Ctrl+Shift+P`).

---

## 🧪 Models Included

-   Logistic Regression
-   K-Nearest Neighbors (KNN)
-   Support Vector Machine (SVM)
-   Decision Tree
-   Random Forest
-   XGBoost
-   Artificial Neural Network (ANN)

---

## 📈 Results & Evaluation

The models were evaluated based on their accuracy on the test set.

| Model                        | Accuracy (%) |
| ---------------------------- | :----------: |
| Naive Bayes                  |    99.64     |
| Random Forest                |    99.45     |
| XGBoost                      |    99.45     |
| Gradient Boosting            |    99.09     |
| Decision Tree                |    98.73     |
| Artificial Neural Network (ANN) |    98.18     |
| Support Vector Machine (SVM) |    98.00     |
| Logistic Regression          |    97.27     |
| K-Nearest Neighbors (KNN)    |    95.82     |

Evaluation is performed using:
* **Accuracy Score**
* **F1 Score** (Macro/Weighted)
* **Confusion Matrix**
* **ROC / AUC Curves** (One-vs-Rest for multi-class classification)

---

## 🗂️ Repository Structure
AgriRecoNet/
├── AgriRecoNet.ipynb
├── requirements.txt
├── .gitignore
└── README.md
---

## 🧭 Roadmap

Future enhancements planned for this project:

-   [ ] Implement hyperparameter tuning sweeps (Optuna/Scikit-Optimize).
-   [ ] Integrate cross-validation with stratification for more robust evaluation.
-   [ ] Add model explainability using **SHAP** across all models.
-   [ ] Create a script to export the best-performing model as a saved artifact for deployment.

---

## 📌 Notes

-   **TensorFlow Errors:** If you encounter GPU errors with TensorFlow, try running on a CPU or ensure you have the correct CUDA/cuDNN versions installed for your hardware.
-   **XGBoost:** This library is optional. If you face installation issues, you can comment out the XGBoost-related cells in the notebook.

---

## 🔑 Keywords

agriculture machine learning, crop recommendation, soil nutrients, nitrogen, phosphorous, potassium, pH, rainfall, temperature, humidity, XGBoost, Random Forest, SVM, KNN, ANN, TensorFlow, Keras, EDA, classification.

