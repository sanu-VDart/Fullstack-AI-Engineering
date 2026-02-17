# 🔧 Mechanical AI Assistant - Turbofan Predictive Maintenance

This project is divided into two main components: a **FastAPI Backend** and a **Next.js Frontend**.

## 🏗️ Project Structure

- `backend/`: Python FastAPI application, ML models, and dataset.
- `frontend/`: Next.js web interface for interaction and visualization.
- `ARCHITECTURE.md`: Detailed system architecture and data flow.

---

## 🚀 Getting Started

### 1. Backend Setup (FastAPI)

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
# Edit .env and add your GOOGLE_API_KEY
# (Already moved from root)

# Start the backend server
python -m uvicorn src.main:app --reload --port 8000
```
API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Frontend Setup (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```
Web Interface: [http://localhost:3000](http://localhost:3000)

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
