# CT Scan Upload Web App

This folder adds a small web UI + API around the existing model in this repo.

## Run

From `lung_cancer_project/`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A (recommended): run without auto-reload (no restart loop)
uvicorn webapp.app:app --port 8000

# Option B: auto-reload, but ONLY watch your project files
uvicorn webapp.app:app --reload --reload-dir webapp --reload-exclude ".venv/*" --port 8000
```

Then open `http://127.0.0.1:8000/`.

## API

- `POST /api/report` (multipart form-data)
  - field: `file` (image)
  - returns: `{ ok: true, report: { predicted_class, confidence, probabilities, device } }`

