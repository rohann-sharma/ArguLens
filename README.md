# ArguLens

ArguLens is an AI-powered Argument Quality, Evidence Verification, and Reasoning Analysis system.

## Setup

### Backend

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # macOS/Linux: source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the API:
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

### Frontend

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
   The dashboard will be available at `http://localhost:5173`.

## Features
- **Claim & Evidence Extraction**: Identifies main arguments and supporting evidence.
- **Source Verification**: Distinguishes between verified, unverified, and non-existent sources.
- **Reasoning Analysis**: Employs NLI to check claim-evidence relationships.
- **Explainable Scoring**: Generates a detailed breakdown of argument quality.

## Sample Data
Check `backend/sample_data/arguments.json` for sample inputs to test.

## Running Tests
In the `backend` folder, run:
```bash
pytest tests/
```
