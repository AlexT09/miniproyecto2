# 🫀 Heart Disease MLOps — Miniproyecto 2

Predicción de enfermedad cardíaca con flujo completo de **Machine Learning Operations (MLOps)** local.

**Dataset:** [Heart Failure Prediction — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**Tarea:** Clasificación binaria — `HeartDisease` (0 = sano, 1 = enfermo)  
**Pacientes:** 918 | **Variables:** 12

---

## 🏗️ Estructura del Proyecto

```
miniproyecto2/
├── app/
│   ├── api.py              ← API REST (FastAPI)
│   └── model.joblib        ← Modelo entrenado (generado por notebook 2)
├── docker/
│   ├── Dockerfile          ← Imagen Docker
│   └── requirements.txt    ← Dependencias Python
├── k8s/
│   ├── deployment.yaml     ← Despliegue Kubernetes
│   └── service.yaml        ← Servicio Kubernetes (LoadBalancer)
├── notebooks/
│   ├── 1_model_leakage_demo.ipynb   ← EDA + Data Leakage + 5 modelos
│   └── 2_model_pipeline_cv.ipynb   ← Modelo final + evaluación + export
├── tests/
│   └── test_api.py         ← Tests unitarios (pytest)
├── .github/workflows/
│   └── ci.yml              ← CI/CD automático con GitHub Actions
├── monitor_drift.py        ← Monitoreo de deriva (Evidently)
├── heart.csv               ← Dataset
└── README.md
```

---

## 🚀 Pasos para Ejecutar

### Requisitos
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib fastapi uvicorn pytest
```

### Etapa 1 y 2 — Notebooks
```
1. Abrir notebooks/1_model_leakage_demo.ipynb → Ejecutar todas las celdas
2. Abrir notebooks/2_model_pipeline_cv.ipynb  → Ejecutar todas las celdas
   (genera app/model.joblib automáticamente)
```

### Etapa 3 — API Local
```bash
uvicorn app.api:app --reload --port 8000
```
- Swagger UI: http://localhost:8000/docs  
- Health check: http://localhost:8000/health  
- Features: http://localhost:8000/features  

**Ejemplo de predicción:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [40, 140, 289, 0, 172, 0.0, 1, 1, 0, 0, 1, 0, 0, 0, 1]}'
```

### Etapa 3 — Docker
```bash
# Construir imagen
docker build -t heart-api -f docker/Dockerfile .

# Ejecutar contenedor
docker run -p 8000:8000 heart-api
```

### Etapa 4 — Kubernetes (Minikube)
```bash
# Subir imagen a Docker Hub primero
docker tag heart-api <TU_USUARIO>/heart-api
docker push <TU_USUARIO>/heart-api

# Reemplazar <TU_USUARIO_DOCKER> en k8s/deployment.yaml, luego:
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get svc
minikube service heart-service
```

### Etapa 5 — CI/CD
El archivo `.github/workflows/ci.yml` ejecuta automáticamente en cada `git push`:
- Lint con flake8
- Tests con pytest
- Build de imagen Docker

### Etapa 6 — Monitoreo
```bash
pip install evidently
python monitor_drift.py
# Abre drift_report.html en el navegador
```

### Tests
```bash
pytest tests/ -v
```

---

## 📊 Modelos Implementados (Notebook 1)

| Modelo | Pipeline | GridSearchCV | Métrica |
|---|---|---|---|
| SVC | ✅ | ✅ | AUC |
| LogisticRegression | ✅ | ✅ | AUC |
| RandomForestClassifier | ✅ | ✅ | AUC |
| KNeighborsClassifier | ✅ | ✅ | AUC |
| GradientBoostingClassifier | ✅ | ✅ | AUC |

## 🔬 Features del Modelo (15 tras encoding)

| # | Feature | Tipo |
|---|---|---|
| 0 | Age | Numérica |
| 1 | RestingBP | Numérica |
| 2 | Cholesterol | Numérica |
| 3 | FastingBS | Binaria |
| 4 | MaxHR | Numérica |
| 5 | Oldpeak | Numérica |
| 6 | Sex_M | Binaria (encoding) |
| 7-9 | ChestPainType_* | Binaria (encoding) |
| 10-11 | RestingECG_* | Binaria (encoding) |
| 12 | ExerciseAngina_Y | Binaria (encoding) |
| 13-14 | ST_Slope_* | Binaria (encoding) |

## 🛠️ Stack Tecnológico

| Componente | Tecnología |
|---|---|
| ML | scikit-learn, pandas, numpy |
| API | FastAPI + Uvicorn |
| Contenedores | Docker |
| Orquestación | Kubernetes (Minikube) |
| CI/CD | GitHub Actions |
| Monitoreo | Evidently |
