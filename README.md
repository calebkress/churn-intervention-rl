# Churn Intervention RL

A full-stack reinforcement learning system that trains a PPO agent to make optimal intervention decisions for telecom customers at risk of churning. Built on MongoDB Atlas — Vector Search, document model, and aggregation pipelines — with a LangChain layer that generates plain-English recommendations grounded in historical customer data.

**Result:** Trained PPO policy reduced simulated churn from 70.5% → 34.5% — a 36-point reduction — across a pool of 10,000 synthetic telecom customers.

---

## What it does

The agent observes a 14-dimensional customer profile (tenure, plan type, spend, call drop rate, NPS, churn probability, and others) and selects one of five interventions: do nothing, send an email offer, make an outbound call, apply a 10% discount, or escalate to a retention specialist. It learns through PPO training to maximize retention while minimizing unnecessary contact costs.

On top of the RL layer, an Atlas Vector Search index enables similarity retrieval — given any customer, the system finds the five most behaviorally similar historical customers, joins their intervention outcomes, and passes that context to `gpt-4o-mini` via a LangChain chain to generate a plain-English recommendation.

---

## Architecture

```
React Dashboard (port 5173)
        │
        ▼
Express (port 3001)
  ├── /api/customers       → Atlas via Mongoose
  ├── /api/interventions   → Atlas via Mongoose
  ├── /api/training-runs   → Atlas via Mongoose
  └── /api/ml/*            → proxy → FastAPI (port 8000)
                                   ├── /infer             ← PPO policy
                                   ├── /insights/:id      ← LangChain
                                   ├── /similar/:id       ← Vector Search
                                   ├── /train
                                   └── /train/status
```

FastAPI owns everything Python — model inference, LangChain chains, Vector Search retrieval. Express owns the application layer and MongoDB queries. React never talks to FastAPI directly.

---

## Stack

| Layer | Technology |
|---|---|
| RL Environment | Python, Gymnasium (custom env) |
| RL Agent | Stable-Baselines3 PPO |
| Data / Vector Search | MongoDB Atlas (M0 free tier) |
| Experiment Tracking | MLflow (local) |
| LangChain | LangChain + OpenAI gpt-4o-mini |
| ML API | FastAPI + Uvicorn |
| Backend API | Node.js + Express + Mongoose |
| Frontend | React + Vite + Recharts |

---

## Repository structure

```
churn-intervention-rl/
├── env/
│   ├── customer.py          # Customer dataclass, churn logic, feature vector
│   ├── reward.py            # Reward function (isolated for tuning)
│   └── churn_env.py         # Gymnasium environment — 14-dim obs, 5 actions
├── data/
│   ├── generate.py          # Synthetic customer generation + Atlas seeding
│   └── atlas.py             # Atlas connection, CRUD, $vectorSearch pipeline
├── agent/
│   ├── train.py             # PPO training loop + MLflow logging
│   └── evaluate.py          # Policy evaluation — writes interventions to Atlas
├── lc/
│   └── insights.py          # LangChain chain: Vector Search → LLM summary
├── ml_api/
│   └── main.py              # FastAPI bridge (port 8000)
├── api/
│   ├── app.js               # Express entry point (port 3001)
│   ├── routes/
│   │   ├── customers.js
│   │   ├── interventions.js
│   │   ├── training_runs.js
│   │   └── insights.js      # proxies /api/ml/* to FastAPI
│   └── db/
│       └── atlas.js         # Mongoose connection
├── client/
│   └── src/
│       ├── App.jsx
│       └── components/
│           ├── RewardCurve.jsx
│           ├── InterventionEffectiveness.jsx
│           ├── SegmentAnalysis.jsx
│           ├── TenureBands.jsx
│           ├── OverIntervention.jsx
│           └── SimilarCustomerInsight.jsx   ← LangChain output
├── models/
│   └── ppo_churn_b4193c90.zip   # canonical trained model
├── requirements.txt
└── package.json
```

---

## Prerequisites

- Python 3.12+
- Node.js 18+
- MongoDB Atlas account (M0 free tier is sufficient)
- OpenAI API key

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/calebkress/churn-intervention-rl
cd churn-intervention-rl

pip install -r requirements.txt
npm install
cd client && npm install && cd ..
```

**2. Configure environment**

Create a `.env` file in the project root:

```
MONGODB_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
MONGODB_DB=churn_rl
FASTAPI_URL=http://localhost:8000
OPENAI_API_KEY=sk-...
PORT=3001
```

**3. Seed Atlas**

```bash
PYTHONPATH=. python data/generate.py
```

Generates 10,000 synthetic telecom customers and writes them to Atlas with pre-computed 10-dimensional feature vectors.

**4. Create the Vector Search index**

In the Atlas UI: your cluster → Atlas Search → Create Search Index → Atlas Vector Search → JSON Editor.

Set index name to `customer_vector_index`, select the `customers` collection, and use this definition:

```json
{
  "fields": [{
    "type": "vector",
    "path": "feature_vector",
    "numDimensions": 10,
    "similarity": "cosine"
  }]
}
```

Wait for status to show **Active** before proceeding.

**5. Run evaluation to populate interventions**

The canonical trained model (`models/ppo_churn_b4193c90.zip`) is included. Run the evaluation to generate the 4,855 intervention records the dashboard visualizes:

```bash
PYTHONPATH=. python agent/evaluate.py
```

---

## Running the application

Three terminals:

```bash
# Terminal 1 — FastAPI (Python ML layer)
PYTHONPATH=. uvicorn ml_api.main:app --reload --port 8000

# Terminal 2 — Express (application layer)
npm run dev

# Terminal 3 — React (dashboard)
cd client && npm run dev
```

Open http://localhost:5173.

---

## RL environment

**Observation space** — `Box(float32, shape=(14,))`:

```
tenure_months_normalized, plan_type_encoded, monthly_spend_normalized,
avg_monthly_data_gb_normalized, call_drop_rate, support_tickets_90d_normalized,
payment_failures_90d_normalized, nps_score_normalized,
days_since_last_contact_normalized, churn_probability,
interventions_this_episode, last_action, steps_remaining_normalized, risk_level
```

**Action space** — `Discrete(5)`:

| Index | Action | Cost |
|---|---|---|
| 0 | `do_nothing` | 0.0 |
| 1 | `send_email_offer` | 0.5 |
| 2 | `outbound_call` | 2.0 |
| 3 | `discount_10pct` | 5.0 |
| 4 | `escalate_to_retention` | 8.0 |

**Episode structure:** 12 timesteps. Terminal when customer churns or episode length is reached. Each reset samples a new customer from the Atlas pool.

**Reward function:**

```python
RETENTION_BONUS      = +15.0   # customer retained this step
CHURN_PENALTY        = -20.0   # customer churned
ACTION_COSTS         = [0, 0.5, 2.0, 5.0, 8.0]  # per action
OVER_CONTACT_PENALTY = -3.0    # > 3 interventions per episode
EFFICIENCY_BONUS     = +1.5    # do_nothing on low-risk customer
REPETITION_PENALTY   = -2.0    # same action 3+ times in a row
```

---

## LangChain integration

`lc/insights.py` implements a retrieval-augmented generation chain:

1. Takes a customer's pre-computed `feature_vector` (10 floats)
2. Runs `$vectorSearch` on Atlas to find the 5 most similar historical customers
3. Joins their intervention history via `$lookup`
4. Formats outcomes as structured context
5. Passes to `gpt-4o-mini` with a telecom retention analyst prompt
6. Returns a 2–3 sentence plain-English recommendation

The Atlas vector index used here is the same index the rest of the application queries — no separate vector store.

---

## Canonical model

| Metric | Value |
|---|---|
| Run ID | `b4193c90` |
| Algorithm | PPO |
| Baseline churn rate | 70.5% |
| Trained churn rate | 34.5% |
| Churn reduction | 36.0 points |
| Mean episode reward | 66.67 |
| Action distribution | outbound_call 52.4% / discount_10pct 47.6% |

---

## Training a new model

```bash
PYTHONPATH=. python agent/train.py
```

Hyperparameters are in `agent/train.py`. Training logs to MLflow locally (`mlruns/`). The new model is written to `models/` and a training run summary is persisted to Atlas.

To use the new model in the API, update `MODEL_PATH` in `ml_api/main.py`.

---

## API reference

**FastAPI (port 8000)**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness — returns model_loaded + db_connected |
| POST | `/infer` | Run policy on 14-dim observation array |
| GET | `/insights/{id}` | LangChain recommendation for a customer |
| GET | `/similar/{id}` | Raw Vector Search results, no LLM |
| POST | `/train` | Trigger background training run |
| GET | `/train/status` | Check training status |

Interactive docs at http://localhost:8000/docs when the server is running.

**Express (port 3001)**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/customers` | Paginated customer list |
| GET | `/api/customers/:id` | Single customer |
| GET | `/api/interventions` | All interventions |
| GET | `/api/interventions/by-customer/:id` | Customer intervention history |
| GET | `/api/training-runs` | All training runs, newest first |
| GET | `/api/training-runs/:id` | Single run with reward_curve |
| GET | `/api/ml/infer/:id` | Build obs from customer → POST to FastAPI |
| GET | `/api/ml/insights/:id` | Proxy to FastAPI /insights |
| GET | `/api/ml/similar/:id` | Proxy to FastAPI /similar |
