# 🧠 Tech Stack Recommender
**DecodeLabs AI Project 3 — Content-Based Filtering Engine**

A recommendation system that maps a user's raw skills to the most aligned career paths and tech stacks using **TF-IDF vectorization** and **Cosine Similarity**.

---

## 📁 Project Structure

```
tech_stack_recommender/
├── recommender.py      # Core engine (TF-IDF + Cosine Similarity logic)
├── app.py              # Streamlit web app
├── raw_skills.csv      # Dataset: 20 job roles with associated skill tags
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the CLI demo
```bash
python recommender.py
```

### 3. Launch the Streamlit web app
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 🔬 Algorithm Overview

### The 4-Step Pipeline

| Step | Stage | Description |
|------|-------|-------------|
| 1 | **Ingestion** | User skills cleaned & joined into a profile string |
| 2 | **Scoring** | TF-IDF vectorization + Cosine Similarity per role |
| 3 | **Sorting** | Roles ranked by similarity score (descending) |
| 4 | **Filtering** | Top-N results returned |

### TF-IDF Weighting
- **TF** (Term Frequency) = how often a skill appears in one role
- **IDF** = `log(total_roles / roles_with_that_skill)` → penalises generic terms
- Result: rare, specific skills (e.g. `Kubernetes`) matter MORE than generic ones (e.g. `Python`)

### Cosine Similarity
```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```
Measures the *angle* between vectors — magnitude-invariant. A role with 20 skills and one with 5 skills are evaluated fairly.

### Cold Start Handling
When input is empty or has zero overlap with dataset vocabulary, the engine falls back to **popularity-based** recommendations (roles with the most diverse skill sets).

---

## 🧪 Example Outputs

**Input:** `["Python", "AWS", "Docker", "Kubernetes", "Automation"]`
```
#1  DevOps Engineer            Score: 0.4707
#2  ML Engineer                Score: 0.2176
#3  Site Reliability Engineer  Score: 0.2061
```

**Input:** `["Python", "Machine Learning", "TensorFlow", "SQL"]`
```
#1  Data Scientist             Score: 0.4509
#2  AI Research Scientist      Score: 0.4346
#3  ML Engineer                Score: 0.3504
```