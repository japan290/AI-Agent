import os
import json
import pandas as pd
import requests
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

# -------------------------------
# Load environment variables from .env or GitHub Secrets
# -------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLICKUP_API_TOKEN = os.getenv("CLICKUP_API_KEY")
TASK_ID = os.getenv("CLICKUP_TASK_ID")

if not all([OPENAI_API_KEY, CLICKUP_API_TOKEN, TASK_ID]):
    raise ValueError("Missing environment variables. Make sure OPENAI_API_KEY, CLICKUP_API_KEY, and CLICKUP_TASK_ID are set.")

# -------------------------------
# Initialize OpenAI client
# -------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# Constants
# -------------------------------
CSV_URL = "https://epoch.ai/data/generated/all_ai_models.csv"
DAYS_TO_FETCH = 20
CACHE_FILE = "last_report.txt"

PROMPT = """
You are a model lifecycle checker.

Task:
- For each model ID, classify it as one of:
  - "Active"
  - "Deprecated but still callable"
  - "Fully Retired / Removed"
- Consider models from OpenAI, Gemini, Claude, DeepSeek, Grok, ElevenLabs, and StabilityAI.
- Use your knowledge as of 2025.
- Respond strictly in JSON with model_id as keys.

Models to classify:
{models}
"""

# -------------------------------
# Functions
# -------------------------------
def read_models_from_file(file_path="models.txt"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

def classify_models(models):
    if not models:
        return {}
    prompt_text = PROMPT.format(models="\n".join(models))
    try:
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a strict JSON classifier for model deprecations."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0
        )
        raw = chat.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:-3]
        return json.loads(raw)
    except Exception as e:
        print(f"⚠️ Error during classification: {e}")
        return {}

def fetch_recent_ai_models(days=DAYS_TO_FETCH):
    try:
        df = pd.read_csv(CSV_URL)
        if 'Publication date' not in df.columns or 'Model' not in df.columns:
            return []
        df['Publication date'] = pd.to_datetime(df['Publication date'], errors='coerce')
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_models = df[df['Publication date'] >= cutoff_date]
        findings = []
        for _, row in recent_models.iterrows():
            model_name = row['Model']
            org = row.get('Organization', 'Unknown')
            pub_date = row['Publication date'].strftime('%Y-%m-%d')
            findings.append({"model": model_name, "org": org, "date": pub_date})
        return findings
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        return []

summary_cache = {}
def get_model_summary(model_name):
    if model_name in summary_cache:
        return summary_cache[model_name]
    prompt = f"Give a one-line description of the AI model '{model_name}' in simple words."
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are an AI assistant giving concise summaries of AI models."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        summary = response.choices[0].message.content.strip().replace("\n", " ")
        summary_cache[model_name] = summary
        return summary
    except Exception:
        return "No summary available."

def group_models(model_statuses):
    groups = {}
    for model, status in model_statuses.items():
        name = model.lower()
        if name.startswith("gpt"):
            group = "OpenAI"
        elif name.startswith("gemini"):
            group = "Gemini"
        elif name.startswith("claude"):
            group = "Claude"
        elif name.startswith("deepseek"):
            group = "DeepSeek"
        elif name.startswith("grok"):
            group = "XAI"
        elif "embedding" in name:
            group = "Text Embedding"
        elif name in ["dalle", "stabilityai", "midjourneyai"]:
            group = "Image Generation"
        elif name.startswith("tts") or name == "elevenlabs":
            group = "Audio Generation"
        else:
            group = "Other"
        groups.setdefault(group, {})[model] = status
    return groups

def update_clickup_task(report_text: str):
    """Replace the description of an existing ClickUp task."""
    url = f"https://api.clickup.com/api/v2/task/{TASK_ID}"  # plain URL
    headers = {"Authorization": CLICKUP_API_TOKEN, "Content-Type": "application/json"}
    payload = {"description": report_text}

    response = requests.put(url, headers=headers, json=payload)
    if response.status_code in [200, 201]:
        print(f"✅ Task {TASK_ID} updated successfully.")
    else:
        print(f"❌ Failed to update task. Status code: {response.status_code}")
        print(response.text)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    models_to_classify = read_models_from_file("models.txt")
    model_statuses = classify_models(models_to_classify)
    new_models = fetch_recent_ai_models(days=DAYS_TO_FETCH)

    # Build report
    report_lines = []
    report_lines.append("📌 **AI Model Lifecycle Report**")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    grouped = group_models(model_statuses)
    for group, models in grouped.items():
        report_lines.append(f"### {group}")
        for model_name, status in models.items():
            report_lines.append(f"- {model_name}: {status}")
        report_lines.append("")

    if new_models:
        report_lines.append("### 🆕 New Models in Last 30 Days")
        for item in new_models:
            model_name = item["model"]
            org = item["org"]
            pub_date = item["date"]
            summary = get_model_summary(model_name)
            report_lines.append(f"- {model_name} ({org}, {pub_date}) → {summary}")

    final_report = "\n".join(report_lines)

    # -------------------------------
    # Compare with cached report
    # -------------------------------
    last_report = ""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            last_report = f.read()

    if final_report.strip() == last_report.strip():
        print("ℹ️ No changes detected. Skipping ClickUp update.")
    else:
        update_clickup_task(final_report)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(final_report)
