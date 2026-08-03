# 🏭 Predictive Maintenance Using Deep Learning (PyTorch)

## 📌 Project Overview
This project implements a custom Feed-Forward Neural Network (FNN) in PyTorch to predict industrial machine failures before they occur. The primary objective is to minimize catastrophic breakdowns, prevent production halts, and optimize maintenance schedules in heavy manufacturing environments.

## 🚀 Key Challenges & Solutions
Working with real-world industrial data comes with unique challenges. Here is how they were solved in this project:

- **Handling Highly Imbalanced Data (97% Normal vs. 3% Failure):** 
  Standard accuracy metrics are useless here. Instead of traditional resampling techniques, I implemented cost-sensitive learning using PyTorch's `BCEWithLogitsLoss` and a dynamically calculated `pos_weight` tensor. This heavily penalized the model for missing actual failures.

- **Domain-Driven Feature Engineering (Mechanical Physics):** 
  Rather than relying purely on raw sensor data, I applied industrial engineering principles to engineer new features that directly correlate with structural breakdowns:
  - `Mechanical Power` (Torque × Rotational Speed): Added to capture conditions leading to mechanical fatigue, creep, and tool wear under extreme load.
  - `Thermal Strain` (Process Temp - Air Temp): Added to identify risks associated with improper heat dissipation and thermal expansion.

- **Architecture Optimization:** 
  Fine-tuned the custom `nn.Module` by integrating `Dropout` layers to prevent overfitting on the minority class and stabilizing the network's learning process.

## 📊 Evaluation & Results
In an industrial setup, a missed failure (False Negative) is far more expensive than a false alarm (False Positive). Therefore, the evaluation strictly focused on the **Recall** metric.

- Successfully achieved a **Recall of ~92%** for the failure class on unseen test data.
- Analyzed performance using Classification Reports and Confusion Matrices to balance the trade-off between True Positives and False Positives.

## 🛠️ Tech Stack
- **Deep Learning Framework:** PyTorch (Custom Architecture, DataLoaders, Optimizers, Backpropagation)
- **Data Manipulation:** Python, Pandas
- **Preprocessing:** Scikit-learn (StandardScaler, Train-Test Split)
- **Visualization:** Matplotlib, Seaborn
