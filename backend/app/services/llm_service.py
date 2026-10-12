import os
import json
import yaml
from pathlib import Path

# Try to import the Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def load_config():
    config_path = BASE_DIR / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def load_prompt():
    prompt_path = BASE_DIR / "prompts" / "argument_reasoning.txt"
    if prompt_path.exists():
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Please evaluate the claims and evidence."

def assess_reasoning(extracted_data: dict) -> dict:
    """
    Calls the Gemini API to perform a claim-evidence reasoning assessment
    based on the extracted NLP data.
    """
    if not SDK_AVAILABLE:
        return {
            "enabled": False,
            "error": "Google GenAI SDK is not installed."
        }

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "enabled": False,
            "error": "GEMINI_API_KEY environment variable not set."
        }

    config = load_config()
    gemini_cfg = config.get("gemini", {})
    if not gemini_cfg.get("enabled", True):
        return {
            "enabled": False,
            "error": "LLM reasoning is disabled in config."
        }

    # Allow environment variable to override config file model
    model_name = os.getenv("GEMINI_MODEL") or gemini_cfg.get("model", "gemini-3.8-flash")
    temperature = gemini_cfg.get("temperature", 0.2)
    
    # Initialize client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return {
            "enabled": False,
            "error": f"Failed to initialize Gemini client: {str(e)}"
        }

    system_instruction = load_prompt()
    
    # Prepare the input payload to the LLM (strip out unnecessary heavy fields if needed)
    payload_to_llm = {
        "claims": extracted_data.get("claims", []),
        "premises": extracted_data.get("premises", []),
        "conclusions": extracted_data.get("conclusions", []),
        "evidence": extracted_data.get("evidence", []),
        "verification_results": [
            {
                "evidence_id": vr.get("evidence_id"),
                "status": vr.get("status"),
                "details": vr.get("details"),
                "source_match_score": vr.get("source_match_score"),
                "numerical_mismatches": vr.get("numerical_mismatches")
            } for vr in extracted_data.get("verification_results", [])
        ],
        "detected_fallacies": extracted_data.get("detected_fallacies", [])
    }
    
    prompt_text = f"Input Argument Data:\n{json.dumps(payload_to_llm, indent=2)}\n"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                response_mime_type="application/json",
            )
        )
        
        result_text = response.text
        # Clean up in case the model returns markdown JSON block
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        parsed_result = json.loads(result_text.strip())
        parsed_result["enabled"] = True
        return parsed_result
        
    except Exception as e:
        return {
            "enabled": False,
            "error": f"LLM assessment failed: {str(e)}"
        }


