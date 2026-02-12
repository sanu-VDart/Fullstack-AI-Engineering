# 🔧 Predictive Maintenance Assistant — Backend

AI-powered predictive maintenance system for turbofan engines using **LangChain**, **LangGraph**, and **Google Gemini**.

## 🚀 Features

- **RUL Prediction** — Predicts Remaining Useful Life using Random Forest
- **State Classification** — Classifies engine state (Normal/Degrading/Critical)
- **AI Chat** — Natural language interface powered by Gemini + LangGraph
- **Maintenance Recommendations** — Automated maintenance scheduling
- **REST API** — FastAPI with full OpenAPI documentation

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| API Framework | FastAPI |
| AI Orchestration | LangChain + LangGraph |
| LLM | Google Gemini 2.5 Flash |
| ML Models | scikit-learn (Random Forest) |
| ML Lifecycle | MLFlow |
| Dataset | NASA C-MAPSS (FD001) |

## 🛠️ Setup

### Prerequisites
- Python 3.11+
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Install & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/MechanicalAI-backend.git
cd MechanicalAI-backend

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Train models (first time only)
python -m src.train_models

# Start the server
uvicorn src.main:app --reload --port 8000
```

### API Docs
Once running, visit: `http://localhost:8000/docs`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/engines` | List available engines |
| POST | `/predict` | Predict RUL for an engine |
| POST | `/chat` | Chat with the AI assistant |
| POST | `/maintenance` | Get maintenance recommendations |

## 🚢 Deployment (Render)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set environment variable: `GOOGLE_API_KEY`
5. Deploy!

## 📊 MLFlow Tracking

```bash
# Run training with MLFlow tracking
python -m src.train_models

# View MLFlow UI
mlflow ui --port 5000
```

## 🏗️ Architecture

```
Client → FastAPI → LangGraph Agent → Gemini 2.5 Flash
                          ↓
                    Tool Execution
                   ┌──────┼──────┐
              Analyze   Predict  Maintain
              Sensors    RUL    Recommend
                   └──────┼──────┘
                     ML Models
                   (scikit-learn)
```

## 📜 License

MIT
