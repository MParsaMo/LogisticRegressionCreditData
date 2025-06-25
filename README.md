# 💳 Credit Default Prediction using Logistic Regression

This project uses a **Logistic Regression** model to predict whether a customer will default on a loan based on features such as income, age, and loan amount. It includes a clean, modular, and well-commented implementation using `pandas` and `scikit-learn`.


---

## 📊 Dataset Format

The project expects a CSV file named `credit_data.csv` with the following columns:

| income | age | loan | default |
|--------|-----|------|---------|
| 50000  | 30  | 10000| 0       |
| 60000  | 45  | 20000| 0       |
| 30000  | 22  | 5000 | 1       |

- `default`: Target column (0 = no default, 1 = default)

> 🔧 If the file does not exist, a dummy dataset will be automatically generated for demonstration purposes.

---

## 🔧 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
