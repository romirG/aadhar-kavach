# UIDAI Datathon - Render Deployment Guide

## 🚀 Quick Deployment Steps

### Step 1: Push Code to GitHub

Make sure all your changes are committed and pushed:

```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### Step 2: Create Services on Render

Go to [Render Dashboard](https://dashboard.render.com) and create **two web services**:

---

## Service 1: ML Backend (Python FastAPI)

1. Click **"New"** → **"Web Service"**
2. Connect your GitHub repository: `UIDAI_DATA_HACKATHON`
3. Configure the service:

| Setting | Value |
|---------|-------|
| **Name** | `uidai-ml-backend` |
| **Root Directory** | `ml_backend` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

4. Add **Environment Variables**:

| Variable | Value |
|----------|-------|
| `PYTHON_VERSION` | `3.10.13` |
| `DATA_GOV_API_KEY` | Your data.gov.in API key |
| `DEBUG` | `false` |

5. Click **"Create Web Service"**

⏳ Wait for this service to deploy before creating the next one!

---

## Service 2: Express Server (Node.js + Frontend)

1. Click **"New"** → **"Web Service"**
2. Connect the same GitHub repository
3. Configure the service:

| Setting | Value |
|---------|-------|
| **Name** | `uidai-express-server` |
| **Root Directory** | `server` |
| **Runtime** | `Node` |
| **Build Command** | `npm install` |
| **Start Command** | `node index.js` |

4. Add **Environment Variables**:

| Variable | Value |
|----------|-------|
| `ML_BACKEND_URL` | Copy the URL of your ML backend (e.g., `https://uidai-ml-backend.onrender.com`) |
| `DATA_GOV_API_KEY` | Your data.gov.in API key |
| `GROQ_API_KEY` | *(Optional)* Your Groq API key for AI features |

5. Click **"Create Web Service"**

---

## 📋 After Deployment

Your services will be available at:

| Service | URL |
|---------|-----|
| **ML Backend** | `https://uidai-ml-backend.onrender.com` |
| **Express + Frontend** | `https://uidai-express-server.onrender.com` |

Open the Express server URL in your browser to see the frontend!

---

## 🔧 Alternative: Blueprint Deployment

If you want to deploy both services at once:

1. Go to Render Dashboard → **"New"** → **"Blueprint"**
2. Connect your repository
3. Render will auto-detect `render.yaml`
4. Click **"Apply"** to create all services

**Note**: You'll still need to manually set the environment variables marked as `sync: false` in the dashboard.

---

## 🐛 Troubleshooting

### "Service not available" / Cold Start

> **Free tier services spin down after 15 minutes of inactivity.**  
> First request after being idle may take 30-60 seconds.

### ML Backend Build Fails

Check for these common issues:
- Ensure `requirements.txt` is in `ml_backend/` folder
- Some packages (like `torch`) may exceed free tier memory - consider Starter plan

### Express Can't Connect to ML Backend

1. Verify `ML_BACKEND_URL` is set correctly in Express service
2. Make sure ML backend is running (check its logs)
3. URL format should be `https://your-ml-backend-name.onrender.com`

### Viewing Logs

1. Go to your service in Render Dashboard
2. Click **"Logs"** tab
3. Click **"Live Tail"** for real-time logs

---

## 💡 Tips

- **Custom Domains**: Add in service Settings → Custom Domains
- **Always-On**: Upgrade to Starter plan ($7/month) to prevent cold starts
- **Scaling**: Render auto-scales based on traffic on paid plans
