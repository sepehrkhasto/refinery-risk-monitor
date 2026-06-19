# Refinery Risk Monitor

AI-powered refinery risk assessment and monitoring platform for
industrial asset health analysis, anomaly detection, and predictive risk
management.

The project combines a modern backend API, machine learning pipelines,
explainable AI techniques, and an interactive monitoring dashboard to
support operational risk analysis in refinery environments.

------------------------------------------------------------------------

## Features

### Risk Assessment

-   Asset risk scoring
-   Operational risk analysis
-   Risk level classification
-   Historical risk tracking

### Machine Learning Pipeline

-   Predictive risk modeling
-   Anomaly detection
-   Trend analysis
-   Root cause analysis
-   Model explainability using SHAP

### Backend Platform

-   FastAPI REST API
-   SQLAlchemy ORM
-   Database migrations with Alembic
-   JWT-based authentication
-   Role-based access control
-   API validation with Pydantic

### Monitoring Dashboard

-   Real-time operational monitoring
-   Risk visualization
-   Sensor data analysis
-   Interactive charts

------------------------------------------------------------------------

# Technology Stack

## Backend

-   Python
-   FastAPI
-   SQLAlchemy
-   Alembic
-   Pydantic
-   JWT Authentication

## Machine Learning

-   Scikit-learn
-   XGBoost
-   LightGBM
-   CatBoost
-   SHAP
-   Pandas
-   NumPy
-   SciPy

## Database

-   SQLite (development)
-   PostgreSQL compatible architecture

------------------------------------------------------------------------

# Installation

## Clone Repository

``` bash
git clone https://github.com/sepehrkhasto/refinery-risk-monitor.git
cd refinery-risk-monitor
```

## Create Virtual Environment

``` bash
python -m venv venv
```

Activate environment and install dependencies:

``` bash
pip install -r requirements.txt
```

## Environment Configuration

Create a `.env` file based on `.env.example`.

Example:

``` env
ENVIRONMENT=development
DATABASE_URL=sqlite:///./refinery_risk.db
SECRET_KEY=your-secret-key
LOG_LEVEL=INFO
```

------------------------------------------------------------------------

# Running the Application

``` bash
uvicorn backend.main:app --reload
```

API documentation:

    http://127.0.0.1:8000/docs

Health check:

    http://127.0.0.1:8000/health

------------------------------------------------------------------------

# Machine Learning Workflow

The ML pipeline supports:

1.  Data preparation
2.  Feature engineering
3.  Model training
4.  Risk prediction
5.  Anomaly detection
6.  Explainability analysis

Generated model artifacts are excluded from Git tracking.

------------------------------------------------------------------------

# Security

Implemented:

-   JWT authentication
-   Password hashing
-   Role-based authorization
-   Environment-based configuration
-   API protection mechanisms

Sensitive files such as `.env`, databases, and trained model artifacts
are excluded from version control.

------------------------------------------------------------------------

# Development

Run tests:

``` bash
pytest
```

Format code:

``` bash
black .
```

Sort imports:

``` bash
isort .
```

Lint:

``` bash
flake8 .
```

------------------------------------------------------------------------

# License

This project is licensed under the MIT License.

------------------------------------------------------------------------

# Author

Refinery Risk Monitor Project

Built with Python, FastAPI, and Machine Learning.
