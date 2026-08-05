"""
============================================================
  Tech Stack Recommender — Core Engine
  DecodeLabs AI Project 3 | Content-Based Filtering
============================================================

HOW IT WORKS (Theory Summary)
──────────────────────────────
1. VECTOR MAPPING
   Machines can't process words like "Python" or "Docker".
   We convert all skill tags into numerical vectors inside a
   shared vocabulary space so math can operate on them.

2. TF-IDF WEIGHTING  (Term Frequency × Inverse Document Frequency)
   - TF  = how often a skill appears in one role's profile
   - IDF = log(total roles / roles containing that skill)
   → Rare, specific skills (e.g. "Kubernetes") get HIGH weight.
   → Common, generic skills (e.g. "Python") get lower weight.
   This prevents generic terms from dominating recommendations.

3. COSINE SIMILARITY
   Measures the ANGLE between the user-profile vector and each
   job-role vector. Magnitude (how many skills) is irrelevant —
   only DIRECTION (what kind of skills) matters.
   Score = (A · B) / (‖A‖ × ‖B‖)
   → 1.0 = perfect alignment | 0.0 = no overlap | -1.0 = opposite

4. THE 4-STEP PIPELINE
   Step 1 — Ingestion : Capture user skills as a space-joined string
   Step 2 — Scoring   : Fit TF-IDF on dataset, transform both sides,
                        compute cosine similarity per role
   Step 3 — Sorting   : Sort roles by similarity score descending
   Step 4 — Filtering : Slice Top-N results (default Top 3)
"""

import os
import math
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────────────────
#  DATASET  —  fallback inline CSV if file not found
# ──────────────────────────────────────────────────────────

INLINE_DATASET = """job_role,skills
Data Scientist,Python SQL Machine_Learning Statistics Data_Analysis NumPy Pandas Visualization TensorFlow Jupyter
ML Engineer,Python TensorFlow PyTorch Machine_Learning Deep_Learning MLOps Docker Kubernetes GPU_Computing Model_Deployment
Backend Developer,Python Java SQL REST_APIs Django Flask Node.js PostgreSQL Redis Microservices
Frontend Developer,JavaScript React Vue CSS HTML UI_UX TypeScript Webpack Figma Accessibility
Full Stack Developer,JavaScript Python React Node.js SQL REST_APIs HTML CSS Docker Git
DevOps Engineer,AWS Docker Kubernetes CI_CD Linux Automation Shell_Scripting Terraform Monitoring Git
Cloud Architect,AWS Azure GCP Cloud Infrastructure Networking Security Automation Terraform Cost_Optimization
Data Engineer,Python SQL ETL Apache_Spark Hadoop Data_Pipelines Airflow Kafka PostgreSQL BigQuery
Cybersecurity Analyst,Networking Security Penetration_Testing Linux Cryptography Firewalls Incident_Response SIEM Compliance
Mobile Developer,Flutter Dart Android iOS React_Native Swift Kotlin Firebase REST_APIs UI_UX
Site Reliability Engineer,Linux Kubernetes Docker Monitoring Automation Python Shell_Scripting CI_CD Incident_Response Observability
Database Administrator,SQL PostgreSQL MySQL MongoDB Oracle Database_Design Query_Optimization Backup_Recovery Performance_Tuning
AI Research Scientist,Python TensorFlow PyTorch Machine_Learning Deep_Learning Research Mathematics NLP Computer_Vision Statistics
Blockchain Developer,Solidity Ethereum Smart_Contracts Web3 Cryptography Python JavaScript Distributed_Systems Security
Game Developer,Unity C# Unreal_Engine C++ Game_Design Physics_Engine Graphics_Programming 3D_Modeling AI_Behavior Optimization
Systems Programmer,C C++ Linux Embedded_Systems Memory_Management Kernel_Development Assembly Performance_Optimization Networking
QA Engineer,Testing Automation Selenium Python Java CI_CD Bug_Tracking Performance_Testing API_Testing Documentation
Product Manager,Agile Product_Strategy User_Research Data_Analysis Communication Roadmapping SQL Stakeholder_Management Scrum
Data Analyst,SQL Python Excel Data_Visualization Tableau Statistics Reporting Business_Intelligence PowerBI Communication
Network Engineer,Networking Cisco Linux Routing_Protocols Firewalls VPN Security TCP_IP Monitoring Troubleshooting
"""


def load_dataset(csv_path: str = "raw_skills.csv") -> pd.DataFrame:
    """
    Load the job roles dataset.
    Falls back to the inline dataset if the CSV file is not found.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from io import StringIO
        df = pd.read_csv(StringIO(INLINE_DATASET))

    # Normalize: lowercase skills, replace underscores for readability later
    df["skills"] = df["skills"].str.lower().str.replace("_", " ")
    df["job_role"] = df["job_role"].str.strip()
    return df


# ──────────────────────────────────────────────────────────
#  STEP 1 — INGESTION
# ──────────────────────────────────────────────────────────

def ingest_user_skills(user_skills: list[str]) -> str | None:
    """
    Step 1: Capture and validate user skill inputs.

    Converts a list like ["Python", "Cloud", "Machine Learning"]
    into a single space-joined lowercase string:  "python cloud machine learning"

    Returns None if input is empty (Cold Start trigger).
    """
    if not user_skills:
        return None  # Cold Start — no profile to match against

    # Normalize: lowercase, strip whitespace, drop empty strings
    cleaned = [s.strip().lower() for s in user_skills if s.strip()]

    if not cleaned:
        return None  # All inputs were blank

    return " ".join(cleaned)


# ──────────────────────────────────────────────────────────
#  STEPS 2, 3, 4 — SCORING → SORTING → FILTERING
# ──────────────────────────────────────────────────────────

def build_tfidf_matrix(df: pd.DataFrame):
    """
    Fit a TF-IDF vectorizer on the job-role skill corpus.

    Why TF-IDF over raw counts?
    → It penalises ubiquitous skills (e.g. "python" appears in 12/20 roles)
      and rewards rare, discriminating ones (e.g. "solidity" appears in 1/20).
    → This produces a more nuanced similarity signal.

    Returns: (vectorizer, tfidf_matrix)
    """
    # ngram_range=(1,2) captures both single-word and two-word skill phrases
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, tfidf_matrix


def score_and_rank(
    user_profile_str: str,
    df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    top_n: int = 3
) -> pd.DataFrame:
    """
    Steps 2 → 3 → 4:

    2. SCORING  — Transform user input using the FITTED vectorizer
                  (same vocabulary space as the job roles),
                  then compute cosine similarity against every role.

    3. SORTING  — Sort results by similarity score descending.

    4. FILTERING — Return only the top_n results.
    """
    # Step 2: Transform user skills into the same TF-IDF vector space
    user_vector = vectorizer.transform([user_profile_str])

    # Cosine similarity: shape → (1, num_roles)
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Attach scores to dataframe
    result_df = df.copy()
    result_df["similarity_score"] = similarity_scores

    # Step 3: Sort descending
    result_df = result_df.sort_values("similarity_score", ascending=False)

    # Step 4: Filter to Top-N
    top_results = result_df.head(top_n).reset_index(drop=True)
    top_results.index += 1  # 1-based ranking for display

    return top_results


# ──────────────────────────────────────────────────────────
#  COLD START HANDLER
# ──────────────────────────────────────────────────────────

def handle_cold_start(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Cold Start Problem: User has no skill history → zero vector → all scores = 0.

    Strategy: Return the most "broadly skilled" roles based on skill count
    (a simple popularity fallback until the user builds a profile).
    This mirrors real-world "Trending Fallback" bypass strategies.
    """
    df = df.copy()
    df["skill_count"] = df["skills"].apply(lambda x: len(x.split()))
    df = df.sort_values("skill_count", ascending=False)
    df["similarity_score"] = 0.0
    result = df.head(top_n)[["job_role", "skills", "similarity_score"]].reset_index(drop=True)
    result.index += 1
    return result


# ──────────────────────────────────────────────────────────
#  PUBLIC API — single entry point
# ──────────────────────────────────────────────────────────

def recommend(
    user_skills: list[str],
    csv_path: str = "raw_skills.csv",
    top_n: int = 3
) -> dict:
    """
    Main recommendation function. Orchestrates the full pipeline.

    Args:
        user_skills : list of skill strings from the user
        csv_path    : path to the skills dataset CSV
        top_n       : number of recommendations to return

    Returns:
        dict with keys:
          "status"       : "success" | "cold_start"
          "results"      : DataFrame with ranked job roles
          "user_profile" : cleaned profile string (or None)
    """
    # Load data and build TF-IDF model
    df = load_dataset(csv_path)
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)

    # Step 1: Ingest
    user_profile = ingest_user_skills(user_skills)

    # Handle Cold Start
    if user_profile is None:
        return {
            "status": "cold_start",
            "results": handle_cold_start(df, top_n),
            "user_profile": None,
        }

    # Steps 2–4: Score, Sort, Filter
    results = score_and_rank(user_profile, df, vectorizer, tfidf_matrix, top_n)

    # Detect if all scores are effectively zero (complete mismatch)
    if results["similarity_score"].max() < 0.01:
        return {
            "status": "cold_start",
            "results": handle_cold_start(df, top_n),
            "user_profile": user_profile,
        }

    return {
        "status": "success",
        "results": results,
        "user_profile": user_profile,
    }


# ──────────────────────────────────────────────────────────
#  CLI DEMO  (run: python recommender.py)
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 55)
    print("  TECH STACK RECOMMENDER  |  DecodeLabs Project 3")
    print("═" * 55)

    test_cases = [
        {
            "label": "Test 1 — Cloud / DevOps profile",
            "skills": ["Python", "AWS", "Docker", "Kubernetes", "Automation"]
        },
        {
            "label": "Test 2 — Data / ML profile",
            "skills": ["Python", "Machine Learning", "TensorFlow", "SQL", "Statistics"]
        },
        {
            "label": "Test 3 — Cold Start (empty input)",
            "skills": []
        },
        {
            "label": "Test 4 — Zero-match skills",
            "skills": ["Cooking", "Dancing", "Pottery"]
        },
    ]

    for test in test_cases:
        print(f"\n▶  {test['label']}")
        print(f"   Input Skills: {test['skills']}")
        output = recommend(test["skills"])

        if output["status"] == "cold_start":
            print("   ⚠  Cold Start detected — showing popular roles as fallback:")
        else:
            print("   ✓  Top Recommendations:")

        for rank, row in output["results"].iterrows():
            score = row["similarity_score"]
            score_str = f"{score:.4f}" if score > 0 else "N/A (fallback)"
            print(f"   #{rank}  {row['job_role']:<30} Score: {score_str}")

    print("\n" + "═" * 55 + "\n")