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
└── ml_models/     the three trained model pipelines (.pkl) + their metrics (.json) — committed
scripts/           training scripts (train_*.py) and rescore_predictions.py
training_data/     where the training CSVs go (downloaded from Kaggle — not committed)
migrations/        Alembic migrations
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

The three trained models are included in `app/ml_models/` (`heart_model.pkl`, `diabetes_model.pkl`, `hypertension_model.pkl`, a few KB each), so the API works straight after cloning — no training step needed. Each sits next to a `*_metrics.json` with its test-set confusion matrix, accuracy/precision/recall/F1/balanced accuracy, and 5-fold cross-validation results.

The training scripts are in `scripts/` for retraining — e.g. after changing a dataset or the preprocessing. The datasets aren't committed; download them from Kaggle and save them in `training_data/` under these names:

| Model | Kaggle dataset | Save as |
|---|---|---|
| Heart disease | [johnsmith88/heart-disease-dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (`heart.csv`, 1,025 rows) | `training_data/Heart_disease.csv` |
| Diabetes | [iammustafatz/diabetes-prediction-dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) (`diabetes_prediction_dataset.csv`, 100,000 rows) | `training_data/Diabetes.csv` |
| Hypertension | [sulianova/cardiovascular-disease-dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) (`cardio_train.csv`, 70,000 rows, `;`-separated) | `training_data/cardio_train.csv` |

Then, from the `backend/` directory:

```bash
python scripts/train_heart_model.py         # training_data/Heart_disease.csv
python scripts/train_diabetes_model.py      # training_data/Diabetes.csv
python scripts/train_hypertension_model.py  # training_data/cardio_train.csv
```

Each script drops duplicate rows, does a stratified 80/20 split with 5-fold cross-validation on the training portion, prints the metrics, and overwrites the model's `.pkl` and `_metrics.json` in `app/ml_models/`. Commit both — the deployed API loads whatever `.pkl` is in the repo.

Dataset quirks the scripts already handle (see each script's comments):

- **Heart** (johnsmith88's Kaggle copy of UCI Cleveland): 723 of its 1,025 rows are exact duplicates (302 unique), and its `target` is **inverted** relative to UCI — `0` means heart disease. It also renumbers `cp`/`restecg`/`slope`/`thal`; `app/services/prediction_service.py` translates the intake form's values to those codes.
- **Diabetes**: 3,854 of its 100,000 rows are exact duplicates (96,146 unique) — mostly rows sharing the BMI fill value 27.32 and the same coarse lab levels, so many may be distinct patients; they're dropped so no identical row lands on both sides of the split.
- **Hypertension** (sulianova): 24 rows are duplicates apart from `id`, the label is derived from the BP readings (≥130/80), implausible readings are dropped, and the model is trained on the other risk factors only.

After retraining, stored predictions were made by the old model — `python scripts/rescore_predictions.py` (dry run by default) re-scores them.

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

Run this against your production database before the first deploy, and after any change to the database tables in `app/models/` (not the ML models). If the database was created by the app on startup rather than by Alembic (no `alembic_version` table), run `alembic stamp ef78ccfb0247` once first.

## Disclaimer

This project generates statistical risk estimates using machine learning models trained on public datasets. It is not a diagnostic tool, has not been clinically validated, and should not be used to make real decisions about anyone's health. If you or someone you know has a health concern, please see an actual healthcare professional.

## License

MIT — see [LICENSE](LICENSE).
