# Air Quality Forecasting Using Diffusion Modeling and ML Algorithms

This project focuses on predicting **PM2.5 concentrations** using a combination of classical machine learning models, deep learning architectures (like LSTM), and an advanced Denoising Diffusion Probabilistic Model (DDPM). The goal is to build a cutting-edge forecasting system that leverages the strengths of generative modeling and multivariate time-series engineering.

---

## Project Overview

- Forecasting air pollution, especially PM2.5, is critical for public health, urban planning, and climate research.
- This project builds a full-stack ML pipeline:
  - Preprocessing raw CSV sensor data
  - Engineering lag and rolling features
  - Training classical regressors (Random Forest, XGBoost)
  - Training a sequence-based LSTM
  - Designing and implementing a custom DDPM to model time series in a novel way
- Visual comparisons and performance benchmarks reveal strengths and limitations across models.

---

## Highlights

- Multivariate Time Series Forecasting  
- Custom Sequence Diffusion Model (DDPM)  
- ML Benchmarking (XGBoost, RF, LSTM)  
- Feature-Rich Lag Analysis  
- Clean Modular Codebase  

---

## Project Structure

air-quality-diffusion/
├── data/ Preprocessed and scaled input datasets
├── models/ Saved model weights (.pth, .json, .pkl)
├── notebooks/ Core notebooks (see below)
├── src/ 
│ └── diffusion_model.py 
├── requirements.txt 
└── README.md 

---

## Notebooks Overview

### 01_explore_clean_data.ipynb

- Loads raw CSV files  
- Handles missing values, interpolation, and scaling  
- Concatenates multivariate air quality features across sensors  
**Output:** `scaled_data.csv`

---

### 02_feature_engineering.ipynb

- Constructs:
  - Lag features (t-1 to t-n)
  - Rolling means and standard deviations
  - Day of week / time encodings  
**Output:** Feature-rich dataset for ML modeling

### 02_dimensionality_rduction.ipynb

- Applies PCA to reduce dimensionality of multivariate feature space
- Compresses high-dimensional sensor data into a lower-dimensional latent space for modeling
- Visualizes variance explained by each principal component and selects optimal number of components
- Prepares compressed input features for use in downstream models (e.g., LSTM, XGBoost, Diffusion)
**Output:** Reduced-dimension dataset and PCA transformation object

---

### 03_lstm_model.ipynb

- Prepares sequential windows for LSTM training  
- Trains an LSTM model for PM2.5 forecasting  
- Plots predictions and error curves  
**Output:** RMSE, R², and predicted vs true PM2.5 plots

---

### 04_diffusion_modeling.ipynb

- Implements a custom `Simple1DDiffusionModel`  
- Trains the DDPM to denoise a latent 1D vector representation of PM2.5 sequences  
- Forecasts via iterative reverse sampling  
- Benchmarks model predictions against ground truth  
**Output:** DDPM performance metrics and forecast visualizations

---

### 05_baseline_models.ipynb

- Trains Random Forest and XGBoost regressors on lag feature data  
- Compares:
  - Random Forest
  - XGBoost
  - LSTM
  - DDPM  
**Output:** Final RMSE, R², and model performance comparison

---

## Tech Stack

- Python 3.10+  
- PyTorch – Deep learning and DDPM modeling  
- XGBoost – Gradient boosting  
- Scikit-learn – Baseline metrics and preprocessing  
- Pandas / NumPy / Matplotlib – Data wrangling and visualization  

---

## How the Diffusion Model Works

- The DDPM is inspired by generative models like DALL·E and Stable Diffusion but adapted for 1D regression.
- Instead of generating images, it learns to reverse noise applied to PM2.5 time series.
- Sampling starts with Gaussian noise and refines over 100+ steps using learned denoising logic.
- The model forecasts future values by iteratively cleaning noisy sequences.

---

## Future Improvements

- Calibrate the diffusion noise schedule (beta_t)  
- Use a Transformer-based denoiser for better attention over time  
- Explore multi-step forecasting (not just one time step ahead)  
- Integrate weather and environmental features (e.g., temperature, humidity)  
- Deploy as a web app using Streamlit or Flask  

---
