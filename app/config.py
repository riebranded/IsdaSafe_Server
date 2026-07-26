import os

# Path to the trained model bundle (.joblib dict with "model"/"features"/"classes" keys).
# Point MODEL_PATH at a different location via an environment variable (e.g. on Render).
MODEL_PATH = os.environ.get("MODEL_PATH", "fish_rf_model.joblib")

# Gemini API key/model used to generate feeding recommendations. Set GEMINI_API_KEY
# via an environment variable (e.g. Render's dashboard) - never commit a real key.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
