# 🧠 Logistic Regression Classification

## 🎯 Objective
The goal of this project is to build a **binary classification model** using **Logistic Regression** to predict whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on various cell nucleus features.  

This task uses the **Breast Cancer Wisconsin Diagnostic Dataset (569 samples, 30 features)**.  
The model learns relationships between numerical measurements and the diagnosis outcome.

---



---

## ⚙️ Libraries & Tools Used
| Category | Libraries |
|-----------|------------|
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib`, `seaborn` |
| **Machine Learning** | `scikit-learn` (`train_test_split`, `LogisticRegression`, `StandardScaler`, `SimpleImputer`, `metrics`) |
| **Environment** | Google Colab or Jupyter Notebook |

---

## 🚀 Steps Performed

### 1️⃣ Load & Inspect Data
- Loaded the dataset (`data.csv`) using **pandas**.
- Checked for null values, datatypes, and descriptive statistics.
- Dropped irrelevant columns (`id`, `Unnamed: 32`).

### 2️⃣ Define Target and Features
- Target variable: **`diagnosis`** (encoded `M=1`, `B=0`)
- Features: all remaining numerical columns (30 features total).

### 3️⃣ Split Dataset
- Train-test split with **80% training** and **20% testing** using `train_test_split` with stratification to preserve class balance.

### 4️⃣ Data Preprocessing
- Handled missing values using **SimpleImputer (mean strategy)**.
- Standardized features using **StandardScaler** for normalization.

### 5️⃣ Train Logistic Regression Model
- Trained **Logistic Regression** using `liblinear` solver with 10,000 iterations.
- Target: predict probability of malignancy.

### 6️⃣ Model Evaluation
Calculated multiple metrics on the test set:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC Curve & AUC**
- **Precision-Recall Curve**
- **Threshold tuning** to visualize trade-offs between precision and recall.

### 7️⃣ Visualization
- Confusion matrix heatmap (Seaborn)
- ROC Curve (TPR vs FPR)
- Precision-Recall Curve
- Sigmoid function visualization
- Feature influence plot (most important variable)



---



## 🧠 Understanding the Model
- **Logistic Regression** models the probability that an observation belongs to class 1 (malignant) using a **sigmoid function**:
  \[
  P(y=1|x) = \frac{1}{1 + e^{-(β_0 + β_1x_1 + ... + β_nx_n)}}
  \]
- The model outputs probabilities between 0 and 1; threshold (default 0.5) decides the class label.
- High **absolute coefficients (|β|)** indicate strong influence on diagnosis prediction.

---

## 🪄 How to Run This Project

### Option 1: Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload:
   - `data.csv`
   - `logistic_regression_breast_cancer.ipynb` (your notebook)
3. Run all cells sequentially.

### Option 2: Local Jupyter Notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
jupyter notebook
