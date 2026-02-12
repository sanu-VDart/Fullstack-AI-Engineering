# ⚛️ AERO-SENSE Diagnostics — Frontend

Cyberpunk-themed predictive maintenance dashboard built with **Next.js**.

## 🚀 Features

- **Real-time Engine Diagnostics** — Engine selection, RUL prediction, state classification
- **AI Chat Console** — Natural language queries about engine health
- **Scrollable Diagnostic Reports** — Complete AI analysis with interactive cards
- **Cyberpunk UI** — Neon-themed dark mode with smooth animations

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Next.js |
| Styling | Custom CSS (cyberpunk theme) |
| Font | Orbitron, Rajdhani |
| API Client | Fetch API |

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/MechanicalAI-frontend.git
cd MechanicalAI-frontend

# Install dependencies
npm install

# Set environment variable
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
# For production, set to your Render backend URL

# Run dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🚢 Deployment (Vercel)

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) → Import Project
3. Connect your GitHub repo
4. Set environment variable: `NEXT_PUBLIC_API_URL` = your Render backend URL
5. Deploy!

## 🔗 Backend

This frontend connects to the [MechanicalAI-backend](https://github.com/YOUR_USERNAME/MechanicalAI-backend).

## 📜 License

MIT
