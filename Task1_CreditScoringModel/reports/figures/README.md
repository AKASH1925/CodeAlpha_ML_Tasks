
---
## 
```markdown
# Model Visualizations & Reports
This directory contains all generated evaluation plots and visualizations for the credit scoring models.
## 📊 Files in This Directory
### 1. **logistic_regression_confusion.png**
**What it shows:**
A confusion matrix for the Logistic Regression model predictions.

**How to read it:**
- **Top-left (TN):** Correctly predicted "Not Creditworthy" ✅
- **Top-right (FP):** Incorrectly predicted "Creditworthy" ❌ (False Alarm)
- **Bottom-left (FN):** Incorrectly predicted "Not Creditworthy" ❌ (Missed Bad Credit)
- **Bottom-right (TP):** Correctly predicted "Creditworthy" ✅
**What to look for:**
- Darker diagonal = better model
- Large FP = too many false alarms
- Large FN = missing actual bad credits (risky!)
---
### 2. **decision_tree_confusion.png**
Same format as above, but for the **Decision Tree** model.
**Comparison:**
- Slightly different accuracy than Logistic Regression
- May have different balance of FP vs FN
---
### 3. **random_forest_confusion.png** 🏆
Same format, but for the **Random Forest** model.
**Key insight:**
Random Forest typically has:
- ✅ Larger TN and TP (more correct predictions)
- ❌ Smaller FP and FN (fewer errors)
---
### 4. **roc_curves.png**
**What it shows:**
ROC (Receiver Operating Characteristic) curves for all 3 models overlaid.
**Visual explanation:**