# Heart Disease Prediction API (FastAPI + Docker)

## 1. Train the model
1. Download the dataset from Kaggle (johnsmith88/heart-disease-dataset) and save it as `data/heart.csv`.
2. Create a virtual env and install the SAME pinned versions the container uses:
   ```
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python train.py
   ```
   This writes `model/heart_model.joblib`.

## 2. Run with Docker
```
docker-compose build
docker-compose up
```
Swagger UI: http://localhost:8000/docs

## 3. Endpoints
- `GET /health`  -> `{"status": "ok"}`
- `GET /info`    -> model type and feature list
- `POST /predict` -> `{"heart_disease": true|false}`

Example request:
```
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":0,"trestbps":150,"chol":270,"fbs":0,"restecg":0,"thalach":111,"exang":1,"oldpeak":0.8,"slope":2,"ca":0,"thal":3}'
```

## 4. Deploy to Render
1. Push this folder to a GitHub repo (commit `model/heart_model.joblib`; `data/` is git-ignored).
2. On render.com: New -> Web Service -> connect the repo.
3. Runtime: **Docker** (Render detects the Dockerfile). Pick the free instance type.
4. Set the Health Check Path to `/health`, then deploy.
5. Your API will be at `https://<service-name>.onrender.com/docs`.
