

#  Formula 1 Podium Prediction System

**End-to-End Machine Learning | FastAPI | Docker | AWS ECS**

A production-style machine learning system that predicts whether a Formula 1 driver will finish on the **podium (Top 3)** in a given race, built with a focus on **real-world ML engineering**, not just model accuracy.

---

##  Project Overview

This project goes beyond notebooks and model training.
It implements a **full ML lifecycle**, from feature engineering and time-aware evaluation to **containerized cloud deployment**.

The system exposes a **public REST API** that serves real-time predictions using a trained XGBoost model, deployed on **AWS ECS (Fargate)** using Docker.

---

##  Problem Statement

Given **pre-race information** about a driver and race context, predict:

```
Will this driver finish on the podium (Top 3)?
```

This is modeled as a **binary classification problem**:

* `1` → Podium finish (P1, P2, P3)
* `0` → Non-podium finish

---

##  Machine Learning Approach

### Model

* **XGBoost Classifier**
* Chosen for:

  * strong performance on tabular data
  * ability to model non-linear interactions
  * widespread industry usage

### Target Variable

```text
podium = 1 if finishing position ∈ {1, 2, 3}, else 0
```

### Features Used
                                                                                       
| Category                 | Features                                                          |
| -------------------------| ------------------------------------------------------------------|                              
| Pre-existing features    | `grid`, `year`, `round`, `circuitId`, `driverId`, `constructorId` |
| Engineered Features from database| `form_last3` , `team_strength`, `age`                     |

Categorical variables (`circuitId`, `driverId`, `constructorId`) are encoded using **OrdinalEncoder**, fitted during training and reused during inference to avoid feature drift.

---

##  Model Evaluation

To simulate real-world deployment, evaluation is **time-aware**:

* **Training data:** up to 2015
* **Test data:** 2016–2023

This prevents future data leakage.

### Performance

             
| Metric   | Score(Using just the 6 pre-existing features) | Score(Using all 9 features) |     
| -------- | ----------------------------------------------|-----------------------------|
| ROC-AUC  | **0.9312**                                    | **0.9410**                  |
| Accuracy | **0.8972**                                    | **0.9084**                  | 

---

##  System Architecture

```text
Raw F1 Data
    ↓
Feature Engineering Pipeline
    ↓
XGBoost Model Training
    ↓
Saved Artifacts (model, encoder, feature list)
    ↓
FastAPI Inference Service
    ↓
Docker Container
    ↓
AWS ECR → AWS ECS (Fargate)
    ↓
Public REST API
```

This mirrors how ML systems are deployed in production environments.

---

##  Project Structure

```text
F1/
├── src/
│   ├── data.py        # Data loading
│   ├── features.py    # Feature engineering
│   ├── train.py       # Model training
│   ├── predict.py     # Inference logic
│   ├── api.py         # FastAPI app
│   └── __init__.py
│
├── data/
│   └── master_features_v1.parquet
│
├── models/
│   ├── model_v1.pkl
│   ├── encoder_v1.pkl
│   └── features_v1.pkl
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── README.md
```

---

##  API Endpoints

### Health Check

```
GET /health
```

Response:

```json
{"status": "ok"}
```

---

### Predict Podium Finish

```
POST /predict
```

**Input**

```json
{
  "grid": 1,
  "driverId": 62,
  "constructorId": 1,
  "year": 2007,
  "round": 24,
  "circuitId": 24,
  "form_last3": 0,
  "age": 40,
  "team_strength": 0
}
```

**Output**

```json
{
  "podium_probability": 0.54,
  "podium_prediction": 1
}
```

---

##  Running Locally with Docker

### Build the image

```bash
docker build -t f1-podium-api .
```

### Run the container

```bash
docker run -p 8080:8080 f1-podium-api
```

### Access API

* Swagger UI: `http://localhost:8080/docs`
* Health check: `http://localhost:8080/health`

---

##  Cloud Deployment (AWS)

* **Container Registry:** AWS ECR
* **Compute:** AWS ECS (Fargate)
* **Networking:** Public IP, custom security group
* **Serving:** FastAPI + Uvicorn

The service is fully containerized and cloud-agnostic, but deployed on AWS ECS (Fargate) to reflect real industry infrastructure and can be started on demand.
A live endpoint can be provided upon request to control cloud costs.


---

##  Engineering Highlights

* Time-aware evaluation to prevent data leakage
* Reusable feature pipeline for training and inference
* Persisted encoders to avoid feature drift
* Dockerized inference for reproducibility
* Cloud-ready deployment using AWS ECS
* Explicit cost-control by stopping services when idle

---





