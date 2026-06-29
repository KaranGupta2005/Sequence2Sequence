# Deployment Guide

## Option 1: Firebase Hosting + Cloud Run (Recommended)

### Step 1: Deploy Backend to Cloud Run

```bash
cd backend

# Build and push Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/seq2seq-backend

# Deploy to Cloud Run
gcloud run deploy seq2seq-backend \
  --image gcr.io/YOUR_PROJECT_ID/seq2seq-backend \
  --platform managed \
  --region us-central1 \
  --memory 512Mi \
  --allow-unauthenticated
```

Note the URL it gives you (e.g., `https://seq2seq-backend-xxxxx.run.app`)

### Step 2: Deploy Frontend to Firebase Hosting

```bash
cd frontend

# Install Firebase CLI
npm i -g firebase-tools
firebase login

# Init (select Hosting, choose your project)
firebase init hosting

# Set your backend URL
# Create .env.local with:
# NEXT_PUBLIC_API_URL=https://seq2seq-backend-xxxxx.run.app

# Build for static export
set FIREBASE_DEPLOY=1 && npx next build
# OR on Mac/Linux: FIREBASE_DEPLOY=1 npx next build

# Deploy
firebase deploy --only hosting
```

### Step 3: Connect Firebase to Cloud Run (optional)

Edit `firebase.json` to proxy `/api/**` to your Cloud Run service. This avoids CORS and hides the backend URL.

---

## Option 1: Docker Compose (Recommended)

```bash
# Make sure weights are in backend/weights/
docker-compose up --build
```
- Frontend: http://localhost:3000
- Backend: http://localhost:8000

## Option 2: Render (Free tier)

### Backend (Render Web Service)
1. Go to render.com → New Web Service
2. Connect your GitHub repo
3. Root Directory: `backend`
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Upload weights to the service disk or use a persistent disk mount

### Frontend (Vercel)
1. Go to vercel.com → Import Project
2. Root Directory: `frontend`
3. Framework: Next.js (auto-detected)
4. Environment Variable: `INTERNAL_API_URL=https://your-backend.onrender.com`
5. Deploy

## Option 3: Railway

```bash
# Install railway CLI
npm i -g @railway/cli
railway login
railway init

# Deploy backend
cd backend
railway up

# Deploy frontend  
cd ../frontend
railway up
```

## Option 4: Manual VPS

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Frontend
cd frontend
npm install
npm run build
npm start
```

## Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `WEIGHTS_DIR` | Backend | Path to model weights folder (default: `weights`) |
| `INTERNAL_API_URL` | Frontend | Backend URL for API proxy (default: `http://127.0.0.1:8000`) |

## Notes

- Model weights (`model.pt` ~100MB) are NOT in git. Upload them separately to your deployment.
- For Render/Railway free tier, the backend cold starts may take 30-60s on first request.
- For Vercel, the frontend is static + serverless rewrites so it's very fast.
