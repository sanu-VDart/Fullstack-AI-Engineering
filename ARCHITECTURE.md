# 🏗️ System Architecture

## High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend - Vercel"
        UI["⚛️ Next.js Dashboard<br/>AERO-SENSE Diagnostics"]
    end
    
    subgraph "Backend - Render"
        API["🚀 FastAPI Server"]
        AGENT["🤖 LangGraph Agent"]
        LLM["💬 Gemini 2.5 Flash"]
        
        subgraph "Tools"
            T1["analyze_sensor_data"]
            T2["predict_rul"]
            T3["get_maintenance_recommendation"]
            T4["list_available_engines"]
        end
        
        subgraph "ML Models"
            RUL["📊 RUL Predictor<br/>Random Forest"]
            STATE["🔧 State Classifier<br/>Random Forest"]
            PREP["⚙️ StandardScaler"]
        end
    end
    
    subgraph "Data & MLOps"
        DATA["📁 NASA C-MAPSS<br/>FD001 Dataset"]
        MLFLOW["📈 MLFlow<br/>Experiment Tracking"]
    end
    
    UI -->|REST API| API
    API --> AGENT
    AGENT -->|Tool Calls| T1 & T2 & T3 & T4
    AGENT <-->|LLM Calls| LLM
    T1 & T2 & T3 --> RUL & STATE & PREP
    RUL & STATE & PREP --> DATA
    MLFLOW -.->|Tracks| RUL & STATE
    
    style UI fill:#1e293b,stroke:#00f3ff,stroke-width:2px
    style API fill:#1e293b,stroke:#00f3ff,stroke-width:2px
    style AGENT fill:#1e293b,stroke:#00f3ff,stroke-width:2px
    style LLM fill:#1e293b,stroke:#a855f7,stroke-width:2px
```

## Request Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend as Next.js<br/>(Vercel)
    participant Backend as FastAPI<br/>(Render)
    participant Agent as LangGraph<br/>Agent
    participant Gemini as Gemini 2.5<br/>Flash
    participant Models as ML Models

    Note over User,Models: Prediction Flow
    User->>Frontend: Select Engine + Predict
    Frontend->>Backend: POST /predict {unit_id}
    Backend->>Models: Load data + predict
    Models-->>Backend: RUL + State + Probabilities
    Backend-->>Frontend: JSON Response
    Frontend-->>User: Display Diagnostic Report

    Note over User,Models: Chat Flow
    User->>Frontend: Ask Question
    Frontend->>Backend: POST /chat {message}
    Backend->>Agent: Process with LangGraph
    Agent->>Gemini: LLM call with tools
    Gemini-->>Agent: Tool call request
    Agent->>Models: Execute tool
    Models-->>Agent: Tool results
    Agent->>Gemini: Feed results back
    Gemini-->>Agent: Generate response
    Agent-->>Backend: Final answer
    Backend-->>Frontend: JSON Response
    Frontend-->>User: Display AI Answer
```

## Deployment Architecture

```mermaid
graph LR
    subgraph "Development"
        DEV_BE["Backend Code"]
        DEV_FE["Frontend Code"]
    end
    
    subgraph "GitHub"
        REPO_BE["MechanicalAI-backend"]
        REPO_FE["MechanicalAI-frontend"]
    end
    
    subgraph "Production"
        RENDER["Render<br/>FastAPI + ML Models"]
        VERCEL["Vercel<br/>Next.js UI"]
        GEMINI["Google Gemini API"]
    end
    
    DEV_BE -->|git push| REPO_BE
    DEV_FE -->|git push| REPO_FE
    REPO_BE -->|Auto Deploy| RENDER
    REPO_FE -->|Auto Deploy| VERCEL
    VERCEL -->|API Calls| RENDER
    RENDER -->|LLM Calls| GEMINI
    
    style RENDER fill:#0f172a,stroke:#00f3ff,stroke-width:3px
    style VERCEL fill:#0f172a,stroke:#00f3ff,stroke-width:3px
    style GEMINI fill:#0f172a,stroke:#a855f7,stroke-width:3px
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js | Cyberpunk UI dashboard |
| **API** | FastAPI | REST endpoints |
| **AI Orchestration** | LangGraph | Agent workflow |
| **LLM** | Gemini 2.5 Flash | Natural language + tool calling |
| **ML Models** | scikit-learn | RUL prediction + state classification |
| **MLOps** | MLFlow | Experiment tracking |
| **Data** | NASA C-MAPSS | Turbofan degradation dataset |
| **Deployment** | Render + Vercel | Backend + Frontend hosting |

---

**Note:** These Mermaid diagrams will render as visual flowcharts when viewed on GitHub.


