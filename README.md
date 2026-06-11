# 🎓 Student Performance Predictor

A machine learning project that predicts whether a student will **Pass or Fail** based on academic and behavioural features using a **Decision Tree Classifier**.

> **Model Accuracy: 100%** on test set *(15-sample demo dataset — see note below)*

---

## 📊 Visualization

![Alt.Text](https://github.com/vanshikaprajapati05/Student-Performance-Predictor/blob/49a992c18b79b6f40a87892c2716e143f3e7ad26/Figure_1.png)

---

## 🔍 Key Findings

- Decision Tree Classifier achieved **100% accuracy** on the test split (3 fail + 2 pass correctly classified)
- **Study Hours** and **Previous Marks** were the strongest predictors of pass/fail outcome
- Students with **Study Hours ≥ 5** and **Attendance ≥ 70%** consistently passed
- Students with **Previous Marks < 50** were correctly classified as at-risk in all cases
- Confusion Matrix: **0 false positives, 0 false negatives** on test data

> ⚠️ Note: The dataset used is a small 15-sample demo dataset created for learning purposes.
> Accuracy of 100% is expected at this scale and would vary on a larger real-world dataset.

---

## 🗂️ Dataset Features

| Feature | Description |
|---|---|
| `StudyHours` | Number of hours studied per day |
| `SleepHours` | Number of hours of sleep per day |
| `Attendance` | Attendance percentage (%) |
| `PreviousMarks` | Marks scored in previous exam |
| `Pass` | Target variable — 1 = Pass, 0 = Fail |

---

## ⚙️ How It Works

```
Raw Data (student_data.csv)
        ↓
Data Exploration (shape, missing values, head)
        ↓
Feature Scaling (StandardScaler)
        ↓
Train/Test Split (80/20)
        ↓
Decision Tree Classifier (scikit-learn)
        ↓
Evaluation (Accuracy + Confusion Matrix)
        ↓
Prediction on new student input
        ↓
Model saved as performance_model.pkl (Joblib)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and exploration |
| Scikit-learn | ML model, StandardScaler, train-test split, metrics |
| Matplotlib | Data visualization |
| Joblib | Saving and loading trained model |

---

## 📁 Project Structure

```
Student-Performance-Predictor/
├── project.py               ← main ML pipeline code
├── student_data.csv         ← dataset (15 student records)
├── performance_model.pkl    ← saved trained Decision Tree model
├── Figure_1.png             ← output visualization
├── requirements.txt         ← Python dependencies
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/vanshikaprajapati05/Student-Performance-Predictor.git
cd Student-Performance-Predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the model**
```bash
python project.py
```

**Expected output:**
```
Accuracy: 1.0
Confusion Matrix:
[[2 0]
 [0 3]]
Prediction for New Student: [1]
Model Saved Successfully
```

---

## 🔮 Future Improvements

- Expand dataset to 500+ real student records for production-level accuracy
- Try Random Forest and Logistic Regression for comparison
- Add cross-validation to reduce overfitting risk on larger datasets
- Build a simple web interface using Streamlit for interactive predictions

---

## 👩‍💻 Author

**Vanshika Prajapati** — B.Sc. Computer Science  
[GitHub](https://github.com/vanshikaprajapati05) · [LinkedIn](https://linkedin.com/in/vanshika-prajapati-655b2939b)


