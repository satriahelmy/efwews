
# ⚡ ML-EFW-EWS

**ML-EFW-EWS: Machine Learning-Based Early Warning System for Electricity Failures Due to Extreme Weather**

This project proposes an Early Warning System (EWS) that leverages machine learning algorithms to predict electricity distribution failures based on extreme weather conditions. Using historical weather data and outage records, the system aims to provide accurate, timely alerts to grid operators and utility companies.

---

## 📌 Overview

ML-EFW-EWS is designed to:
- Detect potential electricity disturbances from meteorological patterns.
- Handle class imbalance using SMOTE, ADASYN, and SMOTE-ENN.
- Apply transformation and robust scaling for outlier-resistant preprocessing.
- Evaluate multiple ensemble models: Random Forest, XGBoost, AdaBoost, and LightGBM.

---

## 📊 System Architecture

```
Weather Data + Outage Labels
        |
     Preprocessing
   (Yeo-Johnson + Scaling)
        |
   Imbalance Handling
  (SMOTE / ADASYN / SMOTE-ENN)
        |
     Model Training
(RandomForest / XGBoost / etc)
        |
   Performance Evaluation
(Accuracy, F1-Score, Confusion Matrix)
        |
   Early Warning Decision
```

---

## 📈 Main Results

The results below represent preliminary performance scores of ensemble models on the classification task.

| Model         | Accuracy | F1-Score | Precision | Recall |
|---------------|----------|----------|-----------|--------|
| Random Forest | 0.87     | 0.81     | 0.84      | 0.79   |
| XGBoost       | 0.88     | 0.82     | 0.85      | 0.80   |
| AdaBoost      | 0.85     | 0.78     | 0.83      | 0.73   |
| LightGBM      | 0.89     | 0.83     | 0.86      | 0.81   |

*Note: Results may vary depending on hyperparameter tuning and resampling strategy used.*

---

## 🚀 Getting Started

### Requirements

- Python 3.8+
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn
- pandas, numpy, matplotlib, seaborn

Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Running the Experiment

1. Place your weather and outage dataset in `/data/`
2. Run the preprocessing and training script:

```bash
python src/train_model.py
```

3. Evaluate results:

```bash
python src/evaluate_model.py
```

---

## 📂 Repository Structure

```
ml-efw-ews/
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for exploration & experiments
├── src/                 # Source code for preprocessing and modeling
├── models/              # Saved models and pipelines
├── requirements.txt     # Python dependencies
└── README.md            # Project overview
```

---

## 📝 Citation

If you use this work or are inspired by it, please consider citing the concept:

```
Helmy Satria Martha Putra. 
"ML-EFW-EWS: Machine Learning-Based Early Warning System for Electricity Failures Due to Extreme Weather", 2025.
```

---

## 📚 Further Reading

- [XGBoost for classification](https://xgboost.readthedocs.io/)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [SMOTE and ADASYN – imbalanced-learn](https://imbalanced-learn.org/stable/)
- [Open-Meteo Weather API](https://open-meteo.com)

---

## 📬 Contact

For questions, ideas, or collaboration:

**Helmy Satria Martha Putra**  
📧 email@example.com  
📍 Indonesia

---

## 🙏 Acknowledgments

- PLN (for internal outage dataset)
- Open-Meteo API (for weather data)
- scikit-learn, imbalanced-learn, and related open-source tools

---

*Built with ⚡ and care using open data and real-world problems.*
