# 🔧 Mechanical AI Assistant - Turbofan Predictive Maintenance

This project is divided into two main components: a **FastAPI Backend** and a **Next.js Frontend**.

## 🏗️ Project Structure

- `backend/`: Python FastAPI application, ML models, and dataset.
- `frontend/`: Next.js web interface for interaction and visualization.
- `ARCHITECTURE.md`: Detailed system architecture and data flow.

---

# 🚀 Getting Started

Since the backend now serves the frontend automatically, you only need to start one service!

### 1. Simple Startup (Recommended)

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the combined server
python -m uvicorn src.main:app --reload --port 8000
```
- **Web Interface**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 2. Development Mode
If you want to edit the frontend code and see changes live:

```bash
# Terminal 1: Backend
cd backend
python -m uvicorn src.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```
(Frontend Dev Server: [http://localhost:3000](http://localhost:3000))

---

## 🛠️ Combined Startup (Windows)

If you are on Windows, you can use the provided script to start both services:

```powershell
./run_app.bat
```

## 📊 Documentation
For more details, see:
- [Architecture Details](ARCHITECTURE.md)
- [API Repair Guide](URGENT_API_KEY_FIX.md)
