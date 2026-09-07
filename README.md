# AI Health Risk Prediction & Telemedicine — Backend

This is the API that powers the AI Health Risk Prediction & Telemedicine platform. It handles patient/doctor/admin authentication, the three machine learning models that generate risk indications for heart disease, diabetes, and hypertension, appointment scheduling, and the real-time chat used during consultations.

The frontend for this project lives in a separate repo: [ai-health-risk-telemedicine-frontend](#) *(update this link once you have it)*.

> This is not a medical device. The risk predictions are statistical estimates from models trained on public datasets, not a diagnosis. See [Disclaimer](#disclaimer).

## Tech stack

- **FastAPI** — the API framework
- **SQLModel** — database models and queries (built on SQLAlchemy)
- **Alembic** — database migrations
- **MySQL** — production database (SQLite works for local development)
- **scikit-learn** — the three Logistic Regression risk models
- **JWT** — authentication, no third-party auth provider

## Project structure

```
app/
├── core/          settings, database session, JWT/password hashing, auth dependencies
├── models/        SQLModel database tables
├── schemas/       request/response validation
├── routers/       auth, predictions, appointments, admin, telemedicine
├── services/      the ML prediction service
└── ml_models/     trained model files (not committed — see "Training the models" below)
training_data/     put your CSV datasets here before training (also not committed)
requirements.txt
.env.example
```

## Running it locally

You'll need Python 3.11+ (this has also been run successfully on 3.14, with a few dependency versions bumped past what you'd get from a plain `pip install` on older guides — see `requirements.txt` for the exact pins that work).

```bash
git clone https://github.com/your-username/your-backend-repo.git
cd your-backend-repo

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` — see the table below for what each value means.

```bash
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` — if the interactive API docs load, you're good.

### Creating the first admin account

There's no public sign-up path for admins, on purpose. Run this instead:

```bash
python -m app.core.bootstrap
```

It'll prompt for an email and password and create an admin account directly.

## Environment variables

| Variable | What it's for | How to get a value |
|---|---|---|
| `DATABASE_URL` | Where the app stores its data | Local dev: `sqlite:///./health_db.sqlite3` (zero setup). Production: a MySQL connection string — `mysql+pymysql://<user>:<password>@<host>:<port>/<database>` |
| `JWT_SECRET_KEY` | Signs login tokens — keep this private | Generate one: `python3 -c "import secrets; print(secrets.token_hex(32))"` |
| `CORS_ORIGINS` | Which frontend URLs are allowed to call this API | `["http://localhost:3000"]` locally; add your deployed frontend's URL (e.g. `https://your-app.vercel.app`) once it's live |
| `ML_MODELS_DIR` | Where the trained model files live | Leave as `app/ml_models` unless you've moved things |

None of these are keys you sign up for anywhere — they're either generated locally or point at a database you control.

## Training the risk models

The three trained model files aren't in this repo — they're a few hundred KB each, and it felt more honest to have people train them on data they've actually inspected rather than ship a black box.

1. Get three datasets — a heart disease dataset, a diabetes dataset, and one with real blood pressure readings for hypertension (a note on this below).
2. Drop the CSVs into `training_data/`.
3. Run:

```bash
python -m app.training.train_heart_model
python -m app.training.train_diabetes_model
python -m app.training.train_hypertension_model
```

Each script handles preprocessing, does a stratified train/test split with cross-validation, prints accuracy/precision/recall/F1/balanced accuracy, and saves the fitted pipeline to `app/ml_models/`.

**Two things worth knowing if you're sourcing your own dataset:**

- **Check for class imbalance before trusting accuracy.** Diabetes datasets in particular tend to be heavily skewed toward the negative class. A model that just predicts "no diabetes" for everyone can look like it's 90% accurate while catching almost none of the actual positive cases. Look at recall, and use `class_weight="balanced"` if needed.
- **Check that the label actually correlates with something.** Not every dataset that claims to predict a condition actually has a learnable signal in it — some synthetic datasets assign the target label independently of the other columns, which no amount of feature engineering can fix. Worth running a quick correlation check between your label and the numeric features before spending time training on it. If you're looking for a hypertension dataset specifically, one reliable approach is to find a dataset with real systolic/diastolic blood pressure readings and derive the "hypertensive" label yourself using a standard clinical threshold, rather than trusting a pre-made label column you can't verify.

## API overview

Full interactive documentation is available at `/docs` once the server is running. Broadly:

- `POST /api/v1/auth/register`, `/login` — account creation and login
- `POST /api/v1/predictions` — submit a health intake form, get back risk indications for all three conditions
- `GET /api/v1/predictions/mine` — a patient's prediction history
- `POST /api/v1/appointments`, `GET /api/v1/appointments/mine` — booking and viewing appointments
- `GET /api/v1/admin/doctors/pending`, `/approve` — doctor approval workflow
- `WS /api/v1/telemedicine/ws/{appt_id}` — real-time consultation chat

## Video calling

The consultation chat runs over a WebSocket built into this API. Video itself is intentionally out of scope here — running that infrastructure well (TURN servers, NAT traversal, recording) is its own project, and managed providers like Daily, Twilio, or Agora handle it far better than a custom build would. Wire your provider of choice into the frontend's consultation page.

## Running tests

```bash
pytest
```

## Deploying

This API needs to run as a persistent process — it holds three ML models in memory and keeps WebSocket connections open — so a serverless platform isn't a great fit. It deploys cleanly to Render, Railway, or Fly.io.

**On Render specifically:**
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Add `DATABASE_URL`, `JWT_SECRET_KEY`, and `CORS_ORIGINS` as environment variables in Render's dashboard
- Once deployed, update `CORS_ORIGINS` to include your actual frontend URL, or the browser will block every request from it

### Database migrations in production

```bash
alembic upgrade head
```

Run this against your production database before the first deploy, and after any change to the models.

## Disclaimer

This project generates statistical risk estimates using machine learning models trained on public datasets. It is not a diagnostic tool, has not been clinically validated, and should not be used to make real decisions about anyone's health. If you or someone you know has a health concern, please see an actual healthcare professional.

## License

MIT — see [LICENSE](LICENSE).
