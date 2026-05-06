# 🫀 Heart Disease MLOps — Miniproyecto 2

**Predicción de enfermedad cardíaca con flujo completo de Machine Learning Operations (MLOps)**

---

## 📌 Descripción del Proyecto

Este libro documenta el desarrollo completo de un sistema de MLOps aplicado a la predicción de enfermedades cardíacas, cubriendo desde la exploración de datos hasta el despliegue en producción con monitoreo continuo.

- **Dataset:** [Heart Failure Prediction — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
- **Tarea:** Clasificación binaria — `HeartDisease` (0 = sano, 1 = enfermo)  
- **Pacientes:** 918 | **Variables:** 12

---

## 🏗️ Arquitectura MLOps

El proyecto implementa un flujo MLOps completo con las siguientes etapas:

| Etapa | Descripción | Tecnología |
|-------|-------------|------------|
| **1. Datos & EDA** | Exploración, preprocesamiento y detección de Data Leakage | pandas, matplotlib, seaborn |
| **2. Modelado** | Pipeline seguro + GridSearchCV + evaluación completa | scikit-learn |
| **3. API** | Servicio REST de predicción en tiempo real | FastAPI + Uvicorn |
| **4. Contenedores** | Empaquetado reproducible | Docker |
| **5. Orquestación** | Despliegue escalable | Kubernetes (Minikube) |
| **6. CI/CD** | Automatización de lint, tests y build | GitHub Actions |
| **7. Monitoreo** | Detección de deriva de datos (data drift) | Evidently |

---

## 📓 Contenido del Libro

### [Etapa 1 — EDA y Data Leakage](notebooks/1_model_leakage_demo)

- Exploración visual del dataset (918 pacientes, 12 variables)
- Encoding de variables categóricas con One-Hot Encoding
- **Demostración de Data Leakage:** flujo incorrecto vs. flujo correcto
- Entrenamiento de 5 clasificadores con Pipeline + GridSearchCV:
  - SVC, Logistic Regression, Random Forest, KNN, Gradient Boosting
- Ranking comparativo por AUC-ROC

### [Etapa 2 — Modelo Final y Exportación](notebooks/2_model_pipeline_cv)

- Reentrenamiento del mejor modelo (GradientBoosting) con búsqueda extendida de hiperparámetros
- Evaluación completa: Matriz de Confusión, Curva ROC, Reporte de Clasificación
- Validación cruzada 10-fold con intervalo de confianza al 95%
- Exportación del modelo como `app/model.joblib` para servir por la API

---

## 🚀 Cómo Ejecutar el Proyecto

### Requisitos
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib fastapi uvicorn pytest evidently jupyter-book
```

### Notebooks
```bash
jupyter notebook notebooks/1_model_leakage_demo.ipynb
jupyter notebook notebooks/2_model_pipeline_cv.ipynb
```

### API Local
```bash
uvicorn app.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### Docker
```bash
docker build -t heart-api -f docker/Dockerfile .
docker run -p 8000:8000 heart-api
```

### Kubernetes (Minikube)
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
minikube service heart-service
```

### Monitoreo de Drift
```bash
python monitor_drift.py
# Abre drift_report.html en el navegador
```

### Construir este Jupyter Book
```bash
jupyter-book build .
# Resultado en: _build/html/index.html
```

---

## 🛠️ Stack Tecnológico

```
scikit-learn · pandas · numpy · FastAPI · Docker · Kubernetes · GitHub Actions · Evidently · Jupyter Book
```

---

*Miniproyecto 2 — Machine Learning Operations (MLOps)*
