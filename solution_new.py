import os
import re
import time
import json
import subprocess
import pandas as pd
import fitz
from groq import Groq

os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

MODEL_ID = "llama-3.1-8b-instant"
REPO_URL = "https://github.com/priyanshu13-stack/ADS_Dataset.git"
LOCAL_REPO_DIR = "./Dataset"
CHECKPOINT_FILE = "processed_files.txt"

# 2. Dataset extraction from github

def fetch_pdfs_from_github(repo_url: str, target_dir: str) -> list:
    """Clones the GitHub repository and locates all PDF files within it."""
    if not os.path.exists(target_dir):
        print(f"Cloning repository from {repo_url}...")
        try:
            subprocess.run(["git", "clone", repo_url, target_dir], check=True, capture_output=True)
            print("Repository cloned successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr.decode()}")
            return []
    else:
        print(f"Local workspace '{target_dir}' already exists. Skipping clone.\n")

    pdf_paths = []
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_paths.append(os.path.join(root, file))

    return sorted(pdf_paths)  # consistent ordering across runs

# ==========================================
# 3. Checkpoint Helpers
# ==========================================
def load_checkpoint() -> set:
    """Returns the set of already-processed filenames."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return {line.strip() for line in f if line.strip()}

def mark_checkpoint(filename: str):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(filename + "\n")

# ==========================================
# 4. Access Quality Scoring Logic
# ==========================================
def calculate_access_score(record: dict) -> int:
    """Calculates an Access Score from 0 to 100."""
    score = 100

    branded_steps = str(record.get("Number of Steps through Brands", "NA")).strip()
    if branded_steps.isdigit() and int(branded_steps) > 0:
        score -= int(branded_steps) * 15

    generic_steps = str(record.get("Number of Steps through Generic", "NA")).strip()
    if generic_steps.isdigit() and int(generic_steps) > 0:
        score -= int(generic_steps) * 10

    if record.get("Step through-Phototherapy", "").upper() == "YES":
        score -= 10

    specialist = str(record.get("Specialist Types", "NA")).strip().upper()
    if specialist not in ["NA", "UNSPECIFIED", "NONE", ""]:
        score -= 10

    init_auth = str(record.get("Initial Authorization Duration(in-months)", "")).strip()
    if init_auth.isdigit() and int(init_auth) < 12:
        score -= 5

    return max(0, min(100, score))

# ==========================================
# 5. LLM Prompt Definition
# ==========================================
SYSTEM_INSTRUCTION = """
You are an expert pharmaceutical access quality analyst.
Identify the pharmaceutical brands covered by the provided Prior Authorization (PA) policy and extract exactly 12 parameters for each brand.

BUSINESS RULES FOR EXTRACTION:
1. Brand: Dynamically identify the specific drug(s)/brand(s).
2. Age: Extract minimum/maximum age. If FDA labelled age is stated, output "FDA labelled age".
3. Step Therapy Requirements Documented in Policy: Capture ALL text related to step therapy.
4. Number of Steps through Brands: Count biologic steps. Take least restrictive path if OR conditions exist. Exclude phototherapy. Output numerical value or 'NA'.
5. Number of Steps through Generic: Count non-biologic/topical steps. Take least restrictive path. Output numerical value or 'NA'.
6. Step through-Phototherapy: 'Yes' if mandatory AND not in an OR statement. Otherwise 'No' or 'NA'.
7. TB Test required: 'Yes' if required, else 'No'.
8. Quantity Limits: Explicit quantity limits only. Output 'NA' if none.
9. Specialist Types: e.g., 'Dermatologist'. Output 'NA' if none.
10. Initial Authorization Duration(in-months): Output in months (e.g., "6", "12") or "Unspecified".
11. Reauthorization Duration(in-months): Output in months (e.g., "6", "12") or "Unspecified".
12. Reauthorization Required: 'Yes' or 'No'.
13. Reauthorization Requirements Documented in Policy: Extract exact text criteria.

OUTPUT FORMAT:
Return ONLY a valid JSON object containing a single key "extracted_policies" which holds an array of objects. Do not include markdown formatting.
"""

# ==========================================
# 6. Smart Document Processing Pipeline
# ==========================================
def extract_and_clean_text(pdf_path: str, max_chars: int = 18000) -> str:
    """Extracts relevant text from PDF, minifies it, and limits size to stay under token limits."""
    text = ""
    keywords = ['psoriasis', 'criteria', 'authorization', 'step', 'therapy', 'approval', 'duration', 'limit']

    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text("text")
            if page.number == 0 or any(k in page_text.lower() for k in keywords):
                text += page_text + " "
        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text[:max_chars].strip()

def process_policy_document(pdf_path: str) -> list:
    """Sends document text to the LLM with backoff/retry logic."""
    filename = os.path.basename(pdf_path)
    policy_text = extract_and_clean_text(pdf_path)

    if not policy_text:
        return []

    max_retries = 3
    current_max_chars = len(policy_text)

    for attempt in range(max_retries):
        truncated_text = policy_text[:current_max_chars]
        prompt = f"Analyze the policy document, identify target brands, and extract parameters.\n\nDOCUMENT TEXT:\n{truncated_text}"

        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            raw_output = response.choices[0].message.content
            try:
                extracted_data = json.loads(raw_output)
            except json.JSONDecodeError as e:
                print(f"JSON parse error for {filename}: {e}")
                break

            results = []
            for record in extracted_data.get("extracted_policies", []):
                record["Filename"] = filename
                record["Access Score"] = calculate_access_score(record)
                if "Brand" in record:
                    record["Brand"] = str(record["Brand"]).upper()
                results.append(record)

            return results

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg:
                wait = 60 * (attempt + 1)  # exponential-ish: 60s, 120s, 180s
                print(f"Rate limit hit for {filename}. Waiting {wait}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            elif "413" in error_msg or "too large" in error_msg:
                current_max_chars = int(current_max_chars * 0.7)
                print(f"Context too large for {filename}. Reducing document to {current_max_chars} chars and retrying...")
            else:
                print(f"API Error processing {filename}: {e}")
                break

    return []

# ==========================================
# 7. Main Execution
# ==========================================
EXPECTED_COLUMNS = [
    "Filename", "Brand", "Age", "Step Therapy Requirements Documented in Policy",
    "Number of Steps through Brands", "Number of Steps through Generic",
    "Step through-Phototherapy", "TB Test required", "Quantity Limits",
    "Specialist Types", "Initial Authorization Duration(in-months)",
    "Reauthorization Duration(in-months)", "Reauthorization Required",
    "Reauthorization Requirements Documented in Policy", "Access Score"
]

def main():
    output_file = "result.csv"

    pdf_paths = fetch_pdfs_from_github(REPO_URL, LOCAL_REPO_DIR)
    if not pdf_paths:
        print("No PDF files found to process. Exiting.")
        return

    already_done = load_checkpoint()
    pending = [p for p in pdf_paths if os.path.basename(p) not in already_done]

    print(f"Found {len(pdf_paths)} PDF(s). {len(already_done)} already processed, {len(pending)} remaining.")

    if not pending:
        print("All files already processed. Results are in result.csv")
        return

    all_extracted_records = []

    for idx, pdf_path in enumerate(pending):
        filename = os.path.basename(pdf_path)
        print(f"Processing ({idx+1}/{len(pending)}): {filename}")

        records = process_policy_document(pdf_path)
        if records:
            all_extracted_records.extend(records)
        mark_checkpoint(filename)

        if idx < len(pending) - 1:
            time.sleep(10)  # respect TPM limit between files

    if all_extracted_records:
        df = pd.DataFrame(all_extracted_records)
        for col in EXPECTED_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        df = df[EXPECTED_COLUMNS]

        # Append to existing CSV if resuming a prior run; otherwise write fresh
        file_exists = os.path.exists(output_file)
        df.to_csv(output_file, mode="a" if file_exists else "w", index=False, header=not file_exists)
        print(f"\nPipeline complete. {len(all_extracted_records)} brand policies saved to {output_file}")
    else:
        print("No data extracted.")

if __name__ == "__main__":
    main()
