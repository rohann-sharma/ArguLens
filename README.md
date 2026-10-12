# ArguLens

ArguLens is an AI-powered Argument Quality, Evidence Verification, and Reasoning Analysis system.

## Pipeline
1. **NLP Extraction:** Extracts claims, premises, conclusions, and cited sources using `spaCy`.
2. **Source Verification:** Automatically queries Crossref, OpenAlex, and PubMed to verify evidence support.
3. **LLM Reasoning Assessment:** Uses Google Gemini (via `google-genai` SDK) to perform a structured claim-evidence reasoning assessment based on the extracted information.
4. **Scoring:** Aggregates metrics to produce an overall quality score.

## LLM Integration
ArguLens integrates with Google Gemini to provide explainable reasoning logic without fabricating facts. The LLM analyzes the argument structure and verification results provided by the earlier pipeline stages.

**To enable the LLM:**
1. Copy `.env.example` to `.env` in the root directory.
2. Set your `GEMINI_API_KEY` in `.env`.
3. (Optional) Set `GEMINI_MODEL` to override the default `gemini-2.5-flash` model.

If the API key is not provided, ArguLens will fall back gracefully to the core NLP scoring and display a warning in the UI.

## How to Run

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --port 8003
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.
