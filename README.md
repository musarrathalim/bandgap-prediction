# 🧪 Predicting Material Bandgaps Using Composition-Based Features

## 📖 Abstract
This project explores whether machine learning models can accurately predict the **bandgap energy** of inorganic compounds using **composition-only features**. Bandgap is a fundamental electronic property of materials, critical for semiconductor and photovoltaic applications. By leveraging **Matminer** feature sets and multiple machine learning algorithms, we aim to understand how feature representation and model selection influence prediction accuracy.

---

## 🎯 Research Question
> **Main Question:**  
> Can we accurately predict the bandgap energy of inorganic compounds using only composition-based features?

### 🔹 Sub-questions
1. How do different composition-based feature sets (e.g., **Magpie** vs **Deml**) affect model performance?  
2. Which machine learning algorithm performs best for bandgap prediction from compositional data?  
3. Does model performance generalize across different datasets (e.g., `matbench_bandgap` vs `matbench_expt_gap`)?

---

## 💡 Motivation
In computational materials science, data-driven methods are becoming essential for discovering new functional materials. However, high-quality crystal structure data is often unavailable, especially for hypothetical compounds. This motivates the need for **composition-based models** that rely only on chemical formulas.  
This project demonstrates:
- How to engineer features from chemical compositions,
- How to compare machine learning models, and
- How to reason about feature effectiveness and model generalization.

---

## 🧰 Tools & Libraries
| Category | Libraries |
|-----------|------------|
| Data Handling | `pandas`, `numpy` |
| Feature Engineering | `matminer`, `pymatgen` |
| Machine Learning | `scikit-learn` |
| Visualization | `matplotlib`, `seaborn` |

---

## 🧩 Dataset Description
We use the **Matbench Bandgap** dataset from the [Matminer](https://hackingmaterials.lbl.gov/matminer/) library:
- 📦 **Name:** `matbench_bandgap`
- 🧱 **Features:** Chemical formula (converted to `Composition`)
- 🎯 **Target:** `band_gap` (in eV)
- 🧹 **Preprocessing:**  
  - Remove missing values  
  - Convert formula → composition  
  - Generate features using Matminer presets  

Optionally, a secondary dataset (`matbench_expt_gap`) can be used to test model generalization.

---

### 🧪 Methodology Overview

1. **Data Collection** – Dataset obtained from Matminer’s materials database (e.g., bandgap dataset).
2. **Data Cleaning** – Removed missing values, standardized formulae.
3. **Feature Engineering** – Applied `ElementProperty` featurizer (Magpie preset) to convert chemical composition into numerical descriptors.
4. **Modeling** – Trained multiple regression models (Random Forest, XGBoost) to predict bandgap energy.
5. **Evaluation** – Compared performance using MAE, RMSE, and R² metrics.
