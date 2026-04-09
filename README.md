# 🏢 Energy Efficiency Prediction for Buildings

> Machine learning system that predicts heating and cooling loads of buildings based on their physical characteristics.

**Bahçeşehir University — Faculty of Engineering and Natural Sciences — Capstone Project**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 About

Buildings account for over 40% of global energy consumption. This project uses machine learning to predict a building's heating and cooling loads from 8 physical characteristics, helping architects and engineers design more energy-efficient structures.

The system trains and compares **10 different ML models**, evaluates them with **8 metrics**, and provides both a **CLI pipeline** and an **interactive web UI** for real-time predictions.

## 🖥️ Screenshots

<img width="1280" height="800" alt="Screenshot_2" src="https://github.com/user-attachments/assets/d99a4545-6221-4b2c-bcf5-ffcc6da26f1b" />
<img width="957" height="317" alt="Screenshot_3" src="https://github.com/user-attachments/assets/7a06d80d-5907-4c8b-a80b-fd17aaa34ce5" />
<img width="956" height="315" alt="Screenshot_4" src="https://github.com/user-attachments/assets/81584104-c375-45dc-8692-0fe30c0b96fc" />
<img width="987" height="593" alt="Screenshot_5" src="https://github.com/user-attachments/assets/70df357b-16bd-447f-b775-3666effb46c5" />

*Run `python app.py` and open http://localhost:5000 to see the interactive prediction dashboard.*

## 🏗️ Project Structure

```
├── app.py                     # Flask web application
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── pipeline/
│   ├── data_loader.py         # Dataset loading & validation
│   ├── preprocessor.py        # Cleaning, scaling, feature selection
│   ├── model_trainer.py       # 10 ML models with hyperparameter tuning
│   ├── evaluator.py           # 8 evaluation metrics + charts
│   ├── predictor.py           # Model persistence & inference
│   └── clustering.py          # K-Means & hierarchical clustering
├── templates/
│   └── index.html             # Web UI
├── utils/
│   └── logger.py              # Logging utility
├── data/                      # Dataset files
├── models/                    # Saved trained models
└── results/                   # Evaluation outputs
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/bunyaminenis/energy-efficiency-prediction-capstone-project.git
cd energy-efficiency-prediction-capstone-project

# Install dependencies
pip install -r requirements.txt
```

### Run the Web App

```bash
python app.py
```

Open **http://localhost:5000** in your browser. The app will auto-train models on the sample dataset and launch the interactive dashboard.

### Run the CLI Pipeline

```bash
# Train all 10 models with full hyperparameter tuning
python main.py --data data/energy_efficiency_sample.csv --mode train --model all

# Train a specific model
python main.py --data data/energy_efficiency_sample.csv --model random_forest

# Predict only heating load
python main.py --data data/energy_efficiency_sample.csv --target heating
```

## 📊 Input Features

| Code | Feature | Unit |
|------|---------|------|
| X1 | Relative Compactness | — |
| X2 | Surface Area | m² |
| X3 | Wall Area | m² |
| X4 | Roof Area | m² |
| X5 | Overall Height | m |
| X6 | Orientation | 2–5 |
| X7 | Glazing Area | ratio |
| X8 | Glazing Area Distribution | 0–5 |

## 🎯 Output Targets

| Code | Target | Unit |
|------|--------|------|
| Y1 | Heating Load | kWh/m² |
| Y2 | Cooling Load | kWh/m² |

## 🤖 ML Models

| # | Model | Type |
|---|-------|------|
| 1 | Linear Regression | Linear |
| 2 | Ridge Regression | Linear (L2 penalty) |
| 3 | Lasso Regression | Linear (L1 penalty) |
| 4 | ElasticNet | Linear (L1+L2) |
| 5 | Decision Tree | Tree-based |
| 6 | Random Forest | Bagging ensemble |
| 7 | Gradient Boosting | Boosting ensemble |
| 8 | Support Vector Regression | Kernel-based |
| 9 | K-Nearest Neighbors | Instance-based |
| 10 | Multi-Layer Perceptron | Neural Network (ANN) |

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MBE | Mean Bias Error |
| CV | Coefficient of Variance |
| MAPE | Mean Absolute Percentage Error |
| MSPE | Mean Squared Percentage Error |
| R² | Coefficient of Determination |

## 📁 Dataset

Compatible with the [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) (768 samples). A synthetic sample dataset is included in `data/` for testing.

## 👥 Team

**Software Engineering**
- A***** F***** (*******)
- Bünyamin Enis Kara (*******)
- M***** T***** (*******)

**Civil Engineering**
- M***** K***** (*******)
- K***** A***** (*******)
- M***** S***** (*******)

**Advisors:** D***** B***** (Software Eng.) · H***** C***** Y***** (Civil Eng.)

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Bahçeşehir University, Faculty of Engineering and Natural Sciences
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/energy+efficiency) for the Energy Efficiency Dataset
- Dr. D***** B***** and Dr. H***** C***** Y***** for their guidance
