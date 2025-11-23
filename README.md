# Linear Regression with OLS, SVD, Gradient Descent and PCA

This project implements and compares several approaches to solving linear regression, using a real-world dataset from `sklearn.datasets`. All core methods are implemented manually in NumPy, without using `sklearn`'s regression models, to emphasize understanding of the underlying linear algebra and optimization.

---

## 1. Project Overview

**Main Goals:**

- Formulate linear regression as a least-squares problem
- Solve it using:
  - **Ordinary Least Squares (OLS)** via the normal equation
  - **SVD-based pseudoinverse**
  - **Batch Gradient Descent**
- Apply **PCA** for dimensionality reduction and rerun regression in the reduced space
- Compare methods in terms of:
  - Train and test error (MSE)
  - Numerical stability (ill-conditioning)
  - Convergence behaviour
  - Effect of dimensionality on performance

**Dataset:**

- `sklearn.datasets.load_diabetes`
- **Task:** Predict a continuous measure of diabetes disease progression
- **Size:** 442 samples, 10 numerical features (age, BMI, blood pressure, etc.)
- All data is loaded directly from `sklearn`; no external files are required

---

## 2. Repository Structure

```
.
├── project.ipynb          # Main notebook with full implementation and results
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── figures/               # Output plots (generated at runtime)
│   ├── ols_residuals_train.png
│   ├── svd_singular_values.png
│   ├── gd_loss_curves.png
│   └── pca_*.png
└── report/
    └── report.pdf         # Final report
```

> **Note:** The core of the project (for grading) is in **`project.ipynb`**, which contains all code, explanations, and numerical results.

---

## 3. Environment and Dependencies

**Developed and tested with:**

- **Python:** 3.10.10
- **Required packages:** `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `jupyter`

### 3.1 Setting up a Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS (bash):**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3.2 Installing Dependencies

Using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy matplotlib pandas scikit-learn jupyter
```

---

## 4. How to Run the Notebook

1. **Activate the virtual environment** (see above)

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

3. **Open `project.ipynb`**

4. **Run all cells** from top to bottom (`Kernel → Restart & Run All`) to:
   - Load and preprocess the dataset
   - Fit models using OLS, SVD, and Gradient Descent
   - Perform PCA and regression in reduced dimensions
   - Generate all plots and quantitative results

The notebook is designed to be **fully reproducible**: running it end-to-end will regenerate all results used in the report.

---

## 5. Implementation Summary

### Task 1 – Dataset Preparation

- Load the diabetes dataset via `load_diabetes()`
- Split into train/test sets using `train_test_split`
- Standardize features using **training-set mean and std only**
- Add a **bias column** to form the design matrices `X_train` and `X_test`

### Task 2 – Ordinary Least Squares (Normal Equation)

- **Implement:** `β̂ = (XᵀX)⁻¹Xᵀy`
- Compute and report **train/test MSE**
- Plot residuals vs. predicted values
- Discuss limitations of the normal equation (singular and ill-conditioned `XᵀX`)

### Task 3 – SVD-Based Solution

- Compute `X = UΣVᵀ` and the pseudoinverse `X⁺ = VΣ⁺Uᵀ`
- **Solve:** `β̂ = X⁺y`
- Compare coefficients and MSE with the OLS solution
- Plot singular values of the design matrix
- Demonstrate robustness on an **ill-conditioned** version of `X` (with near-duplicate features)

### Task 4 – Gradient Descent

- **Define MSE loss:** `L(β) = (1/2n)||Xβ - y||²`
- Implement batch Gradient Descent with gradient: `∇_β L(β) = (1/n)Xᵀ(Xβ - y)`
- Experiment with different learning rates
- Plot **loss vs. iterations** for each learning rate
- Compare the converged GD solution with the analytical OLS/SVD solution (coefficients and MSE)

### Task 5 – PCA and Dimensionality Reduction

- Use SVD on the standardized training data to obtain principal components
- Compute **explained variance ratio** and **cumulative variance**
- Project data onto the top-k PCs and run regression for each k
- Plot **MSE vs. k** and compare with full-dimensional OLS
- Discuss the trade-off between dimensionality, variance explained, and model performance

---

## 6. Reproducibility and Evaluation

- All random operations (e.g., train/test split) use a fixed `random_state`/`seed` for reproducibility
- The notebook computes:
  - Train/Test MSE for each method
  - Norm differences between solutions (e.g., OLS vs SVD, OLS vs GD)
  - PCA-based performance across different numbers of components
- These values are reported directly in the final PDF report

---

## 7. Key Insights

This project demonstrates:

- **From-scratch implementations** (NumPy only) highlighting:
  - Linear algebra formulations (normal equation, SVD)
  - Numerical stability and ill-conditioning challenges
  - Optimization via gradient descent
  - PCA as a dimensionality reduction tool prior to regression
  
- **Practical comparisons** showing:
  - When OLS fails due to ill-conditioning
  - How SVD provides robust solutions
  - The impact of learning rate on gradient descent convergence
  - The bias-variance trade-off in dimensionality reduction

---

## 8. Output Figures

The `figures/` directory stores plots generated by the notebook:

- `ols_residuals_train.png` – Residual analysis for OLS
- `ols_univariate_bmi.png` – Single feature regression visualization
- `svd_singular_values.png` – Singular value spectrum
- `gd_loss_curves.png` – Gradient descent convergence
- `pca_explained_variance.png` – Scree plot
- `pca_cumulative_variance.png` – Cumulative variance explained
- `pca_error_vs_k.png` – MSE vs. number of components
- `pca_2d_scatter.png` – Data visualization in PC space (optional)

All figures are automatically saved after running all cells in `project.ipynb`.

---

## 9. Notes for Instructors

- No external data files are required; the entire experiment is self-contained
- All implementations avoid `sklearn`'s built-in regression models
- The notebook is fully reproducible by simply running `project.ipynb` after installing dependencies
- Mathematical derivations and implementation details are documented within the notebook cells

---

## License

This project is for educational purposes as part of a course project.