
## Project Overview

Network intrusion detection systems (IDS) play a critical role in identifying and mitigating cyber threats in real time. However, modern network traffic datasets—such as CIC-IDS 2017—often suffer from severe class imbalance, where benign traffic vastly outnumbers malicious samples. This skew can lead to high false-negative rates for rare attack types.

In this project, we propose a **hybrid IDS pipeline** that integrates:

- **Data preprocessing** with Random Forest–based feature reduction  
- **Difficult Set Sampling Technique (DSSTE)** for strategic oversampling/undersampling  
- Four deep-learning and ensemble models (XGBoost, LSTM, MiniVGGNet, AlexNet)  
- **Hyperparameter optimization** (GridSearchCV for tree-based models; Keras Tuner for neural networks)  

Our optimized models achieve **>99% accuracy** on most attack classes, demonstrating robustness even under heavy class imbalance.  

---

## Key Features & Contributions

1. **Feature Reduction via Random Forest**  
   - Identify and retain top predictive features from the CIC-IDS 2017 dataset.  
   - Reduce dimensionality while preserving essential information.

2. **Difficult Set Sampling Technique (DSSTE)**  
   - Partition data into “easy” vs. “difficult” samples using KNN and Random Forest.  
   - Apply K-means clustering on majority-class “difficult” subsets to compress and undersample.  
   - Augment minority classes via targeted oversampling, ensuring balanced representation.

3. **Four Model Architectures**  
   - **XGBoost 2**: Gradient-boosted decision tree with tuned parameters (n_estimators, learning_rate, colsample_bytree).  
   - **LSTM 3**: Recurrent neural network with tuned layers, units, and dropout.  
   - **MiniVGGNet 2**: Convolutional network with optimized kernel sizes and dense-layer units.  
   - **AlexNet 2**: Deep CNN with tuned filters and activation functions.

4. **Hyperparameter Optimization**  
   - **Tree-based models** (XGBoost, MiniVGGNet) use **GridSearchCV** for exhaustive search over candidate hyperparameters.  
   - **Neural networks** (LSTM, AlexNet) leverage **Keras Tuner** (RandomSearch) to efficiently explore layer configurations, dropout rates, and learning rates.

5. **State-of-the-Art Performance**  
   - XGBoost 2: 99.20% accuracy | Precision 0.9941 | Recall 0.9920 | F1 0.9926  
   - LSTM 3: 99.06% accuracy | Precision 0.9916 | Recall 0.9979 | F1 0.9948  
   - MiniVGGNet 2: 99.17% accuracy | Precision 0.9920 | Recall 0.9988 | F1 0.9954  
   - AlexNet 2: 91.66% accuracy | Precision 0.9362 | Recall 0.9903 | F1 0.9625  

6. **Comprehensive Analysis & Visualization**  
   - Confusion matrices for each model highlight per-class detection rates.  
   - Training vs. validation loss/accuracy plots to verify convergence and overfitting behavior.

---

## Technology Stack

- **Programming Language**: Python 3.8+  
- **Machine Learning Libraries**:  
  - TensorFlow 2.x / Keras (for LSTM, AlexNet, MiniVGGNet)  
  - XGBoost (for gradient-boosted trees)  
  - scikit-learn (for RandomForest, GridSearchCV, KNN, K-means, evaluation metrics)  
  - Keras Tuner (for neural network hyperparameter search)  
- **Data Handling & Visualization**:  
  - pandas, NumPy, Matplotlib, Seaborn  
- **Environment**:  
  - Google Colab (GPU-enabled), or local machine with NVIDIA GPU support for faster training.

---

## Methodology Highlights

1. **Feature Reduction (Random Forest Regressor)**  
   - Ranked features by importance (e.g., Average Packet Size, Flow Packets/s, Subflow Bwd Bytes, etc.).  
   - Selected top-10 predictive features to simplify the dataset.

2. **Difficult Set Sampling Technique (DSSTE)**  
   - **Step 1**: Partition the dataset into “easy” samples (where K-nearest neighbors share the same class) vs. “difficult” samples.  
   - **Step 2**: For majority-class “difficult” samples, perform K-means clustering to obtain cluster centroids, replacing noisy or redundant points.  
   - **Step 3**: For minority-class “difficult” samples, generate synthetic points by slightly adjusting continuous attributes within a zoom range [1-1/L, 1+1/L].  
   - Combine easy samples, compressed majority samples, original minority samples, and new synthetic minority samples into the final balanced training set.

3. **Model Architectures & Hyperparameters**  
   - **XGBoost 2**: Tuned `n_estimators` ∈ [100,200], `learning_rate` ∈ [0.01,0.1], `colsample_bytree` ∈ [0.6,0.8].  
   - **LSTM 3**: Tuned layers = 2–4, units per layer ∈ [64,128,256], dropout ∈ [0.2,0.5].  
   - **MiniVGGNet 2**: Tuned convolutional filters, kernel sizes ∈ {(3,3),(5,5)}, dense-layer units ∈ [64,128], learning rates ∈ [1e-4,1e-3].  
   - **AlexNet 2**: Tuned number of filters per conv block, activation functions (ReLU vs. LeakyReLU), and dropout rates.

4. **Performance Evaluation**  
   - Four standard metrics: Accuracy, Precision, Recall, F1-Score.  
   - Confusion matrices to reveal per-class strengths/weaknesses.  
   - Training vs. validation loss/accuracy curves to detect overfitting or underfitting.

---

## Results Summary

| Model           | Accuracy | Precision | Recall  | F1-Score |
| --------------- | -------- | --------- | ------- | -------- |
| **XGBoost 2**   | 99.20%   | 0.9941    | 0.9920  | 0.9926   |
| **LSTM 3**      | 99.06%   | 0.9916    | 0.9979  | 0.9948   |
| **MiniVGGNet 2**| 99.17%   | 0.9920    | 0.9988  | 0.9954   |
| **AlexNet 2**   | 91.66%   | 0.9362    | 0.9903  | 0.9625   |

> **Note**: XGBoost 2 and MiniVGGNet 2 both exceed 99% accuracy, outperforming prior benchmarks on the CIC-IDS 2017 dataset. AlexNet 2 shows substantial improvement over its previous iteration, with a 1% accuracy gain and a marked increase in F1-Score.



---



*Thank you for exploring our Hybrid IDS project! We hope this README gives a clear, professional overview that highlights our technical depth, practical impact, and collaboration skills.*  
