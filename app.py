"""
============================================================
  Tech Stack Recommender — Streamlit Web App
  DecodeLabs AI Project 3
  Run: streamlit run app.py
============================================================
"""

import streamlit as st
import pandas as pd
import os, sys

# Ensure recommender module is importable from the same folder
sys.path.insert(0, os.path.dirname(__file__))
from recommender import recommend, load_dataset

# ──────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tech Stack Recommender",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────
#  CUSTOM CSS  —  dark terminal-meets-blueprint aesthetic
# ──────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Base ─────────────────────────────── */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #c9d1d9;
  }

  /* ── Header ────────────────────────────── */
  .hero-header {
    background: linear-gradient(135deg, #161b22 0%, #1a2332 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
  }
  .hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #f0f6fc;
    letter-spacing: -0.5px;
    margin: 0 0 0.35rem;
  }
  .hero-subtitle {
    color: #8b949e;
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }
  .badge {
    display: inline-block;
    background: rgba(59,130,246,0.15);
    border: 1px solid #3b82f6;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-top: 8px;
  }
  .badge.purple { background: rgba(139,92,246,0.15); border-color: #8b5cf6; color: #a78bfa; }
  .badge.cyan   { background: rgba(6,182,212,0.15);  border-color: #06b6d4; color: #22d3ee; }

  /* ── Sidebar ────────────────────────────── */
  [data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
  }
  .sidebar-section {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1rem 0.5rem;
    margin-bottom: 1rem;
  }
  .sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
  }

  /* ── Cards ──────────────────────────────── */
  .rank-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1.3rem 1.4rem;
    margin-bottom: 0.85rem;
    position: relative;
    transition: border-color 0.2s;
  }
  .rank-card.rank-1 { border-left: 4px solid #3b82f6; }
  .rank-card.rank-2 { border-left: 4px solid #8b5cf6; }
  .rank-card.rank-3 { border-left: 4px solid #06b6d4; }
  .rank-card.rank-4 { border-left: 4px solid #6b7280; }
  .rank-card.rank-5 { border-left: 4px solid #6b7280; }

  .rank-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
  }
  .role-name {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f0f6fc;
    margin: 0.2rem 0 0.5rem;
  }
  .score-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin: 0.4rem 0;
  }
  .score-bar-fill {
    height: 6px;
    border-radius: 4px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  }
  .score-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #60a5fa;
    font-weight: 600;
  }
  .skill-pill {
    display: inline-block;
    background: #21262d;
    border: 1px solid #30363d;
    color: #8b949e;
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 2px 2px 0;
    font-family: 'JetBrains Mono', monospace;
  }
  .skill-pill.matched {
    background: rgba(59,130,246,0.12);
    border-color: #3b82f6;
    color: #60a5fa;
  }

  /* ── Pipeline steps ─────────────────────── */
  .pipeline-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.6rem;
    margin: 1rem 0;
  }
  .pipeline-step {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 0.7rem;
    text-align: center;
  }
  .step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
  }
  .step-label {
    font-size: 0.72rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .s1 { color: #3b82f6; }
  .s2 { color: #8b5cf6; }
  .s3 { color: #06b6d4; }
  .s4 { color: #10b981; }

  /* ── Alert boxes ────────────────────────── */
  .cold-start-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    color: #fbbf24;
    font-size: 0.88rem;
  }
  .success-box {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 8px;
    padding: 0.75rem 1.2rem;
    margin-bottom: 1rem;
    color: #34d399;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Misc ───────────────────────────────── */
  .stButton > button {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.5rem;
    font-weight: 600;
    font-size: 0.9rem;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }
  div[data-testid="stTextInput"] input {
    background: #21262d;
    border: 1px solid #30363d;
    color: #f0f6fc;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
  }
  .stMultiSelect [data-baseweb="tag"] {
    background-color: rgba(59,130,246,0.2) !important;
    color: #60a5fa !important;
  }
  h2, h3 { color: #f0f6fc !important; }
  hr { border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  HELPER — render a result card
# ──────────────────────────────────────────────────────────

def render_result_card(rank: int, role: str, skills_str: str,
                       score: float, user_skills_lower: list[str]):
    colors = {1: "#3b82f6", 2: "#8b5cf6", 3: "#06b6d4", 4: "#6b7280", 5: "#6b7280"}
    color = colors.get(rank, "#6b7280")
    bar_width = int(score * 100)

    # Build skill pills with matched ones highlighted
    role_skills = skills_str.split()
    pills = ""
    for skill in role_skills[:12]:  # cap display at 12
        display = skill.replace("_", " ").title()
        matched = skill.replace("_", " ") in user_skills_lower or skill in user_skills_lower
        cls = "skill-pill matched" if matched else "skill-pill"
        pills += f'<span class="{cls}">{display}</span>'
    if len(role_skills) > 12:
        pills += f'<span class="skill-pill">+{len(role_skills)-12} more</span>'

    score_display = f"{score:.4f}" if score > 0 else "N/A"

    st.markdown(f"""
    <div class="rank-card rank-{rank}">
      <div class="rank-number">#{rank} match</div>
      <div class="role-name">{role}</div>
      <div class="score-bar-bg">
        <div class="score-bar-fill" style="width:{bar_width}%; background: linear-gradient(90deg, {color}, {color}aa);"></div>
      </div>
      <div class="score-value">cosine similarity: {score_display}</div>
      <div style="margin-top:0.7rem;">{pills}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
      <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#f0f6fc; font-weight:700;">
        🧠 TechMatch
      </div>
      <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:1px;">
        AI Recommender · v1.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">⚙ Configuration</div>', unsafe_allow_html=True)

    top_n = st.slider("Top-N Results", min_value=1, max_value=10, value=3, step=1)

    st.markdown('</div>', unsafe_allow_html=True)

    # Available skills reference
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">📚 Skill Reference</div>', unsafe_allow_html=True)

    df_ref = load_dataset(
        os.path.join(os.path.dirname(__file__), "raw_skills.csv")
    )
    all_skills = sorted(set(
        skill.strip().replace("_", " ").title()
        for row in df_ref["skills"]
        for skill in row.split()
    ))
    st.caption(f"{len(all_skills)} indexed skills")
    skill_display = " · ".join(all_skills[:40]) + "..."
    st.markdown(f'<div style="font-size:0.72rem; color:#8b949e; line-height:1.8;">{skill_display}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Dataset info
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">🗄 Dataset Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.78rem; color:#8b949e;">
      <b style="color:#c9d1d9;">Roles:</b> {len(df_ref)}<br>
      <b style="color:#c9d1d9;">Algorithm:</b> TF-IDF + Cosine Similarity<br>
      <b style="color:#c9d1d9;">Source:</b> raw_skills.csv
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem; font-size:0.7rem; color:#484f58; text-align:center; line-height:1.8;">
      DecodeLabs · AI Project 3<br>
      Content-Based Filtering
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  MAIN CONTENT — HERO
# ──────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
  <div class="hero-title">⚡ Tech Stack Recommender</div>
  <div class="hero-subtitle">AI Recommendation Logic · Pattern Alignment Engine</div>
  <div style="margin-top:0.8rem;">
    <span class="badge">TF-IDF Vectorization</span>
    <span class="badge purple">Cosine Similarity</span>
    <span class="badge cyan">Content-Based Filtering</span>
  </div>
</div>
""", unsafe_allow_html=True)

# 4-step pipeline visual
st.markdown("""
<div class="pipeline-grid">
  <div class="pipeline-step"><div class="step-num s1">01</div><div class="step-label">Ingestion</div></div>
  <div class="pipeline-step"><div class="step-num s2">02</div><div class="step-label">Scoring</div></div>
  <div class="pipeline-step"><div class="step-num s3">03</div><div class="step-label">Sorting</div></div>
  <div class="pipeline-step"><div class="step-num s4">04</div><div class="step-label">Filtering</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────
#  INPUT SECTION
# ──────────────────────────────────────────────────────────

col_input, col_results = st.columns([1, 1.4], gap="large")

with col_input:
    st.markdown("### 🎯 Your Skill Profile")
    st.caption("Enter your technical skills below. The engine maps these to job roles using vector similarity.")

    # Multi-select from known skills
    selected_from_list = st.multiselect(
        "Pick from indexed skills",
        options=all_skills,
        placeholder="Search & select skills...",
        help="These are skills already in the dataset vocabulary."
    )

    # Free-text input for custom skills
    custom_input = st.text_input(
        "Or type custom skills (comma-separated)",
        placeholder="e.g.  GraphQL, Rust, WebAssembly",
        help="Type skills not in the list above — they'll be matched by TF-IDF if they overlap with role vocabulary."
    )

    # Quick presets
    st.caption("🚀 Quick presets:")
    preset_col1, preset_col2 = st.columns(2)
    preset = None
    with preset_col1:
        if st.button("☁ Cloud / DevOps"):
            preset = ["Python", "AWS", "Docker", "Kubernetes", "Automation", "Linux", "Terraform"]
        if st.button("🤖 AI / ML"):
            preset = ["Python", "Machine Learning", "TensorFlow", "Statistics", "Deep Learning", "NLP"]
    with preset_col2:
        if st.button("🖥 Backend Dev"):
            preset = ["Python", "Java", "SQL", "REST APIs", "PostgreSQL", "Microservices"]
        if st.button("📊 Data Analyst"):
            preset = ["SQL", "Python", "Excel", "Tableau", "Statistics", "Data Visualization"]

    st.markdown("---")

    run_button = st.button("🔍  Find My Tech Stack")

# ──────────────────────────────────────────────────────────
#  RECOMMENDATION LOGIC + RESULTS
# ──────────────────────────────────────────────────────────

with col_results:
    st.markdown("### 📊 Recommendations")

    # Build final skill list
    final_skills = list(selected_from_list)

    if preset:
        final_skills = preset
        st.info(f"Preset loaded: `{', '.join(preset)}`")

    if custom_input.strip():
        custom_skills = [s.strip() for s in custom_input.split(",") if s.strip()]
        final_skills += custom_skills

    if run_button or preset:
        if not final_skills:
            st.markdown("""
            <div class="cold-start-box">
              ⚠️  <b>Cold Start Detected</b><br>
              No skills provided. Please select or type at least one skill to personalise recommendations.
              Showing popular roles as fallback.
            </div>
            """, unsafe_allow_html=True)

        with st.spinner("Running similarity engine..."):
            output = recommend(
                user_skills=final_skills,
                csv_path=os.path.join(os.path.dirname(__file__), "raw_skills.csv"),
                top_n=top_n
            )

        if output["status"] == "cold_start" and final_skills:
            st.markdown("""
            <div class="cold-start-box">
              ⚠️  <b>Zero-Match Cold Start</b><br>
              Your skills don't overlap with any role's vocabulary. Showing popularity-based fallback.
              Try skills like: Python, SQL, AWS, Docker, JavaScript…
            </div>
            """, unsafe_allow_html=True)
        elif output["status"] == "success":
            user_lowers = [s.lower().replace("_", " ") for s in final_skills]
            matched_skills_str = ", ".join(f"`{s}`" for s in final_skills[:8])
            st.markdown(f"""
            <div class="success-box">
              ✓ Profile ingested → {len(final_skills)} skill(s) mapped to vector space<br>
              ✓ Cosine similarity computed across {len(df_ref)} job roles<br>
              ✓ Top {top_n} results filtered
            </div>
            """, unsafe_allow_html=True)

        # Render result cards
        results_df = output["results"]
        user_skills_lower = [s.lower() for s in final_skills]

        for rank, row in results_df.iterrows():
            render_result_card(
                rank=int(rank),
                role=row["job_role"],
                skills_str=row["skills"],
                score=float(row["similarity_score"]),
                user_skills_lower=user_skills_lower
            )

        # Similarity score table (expandable)
        with st.expander("📋 View full similarity score table"):
            display_df = results_df[["job_role", "similarity_score"]].copy()
            display_df.columns = ["Job Role", "Cosine Similarity Score"]
            display_df["Cosine Similarity Score"] = display_df["Cosine Similarity Score"].map(
                lambda x: f"{x:.6f}"
            )
            st.dataframe(display_df, use_container_width=True)

    else:
        st.markdown("""
        <div style="
          background: #161b22;
          border: 1px dashed #30363d;
          border-radius: 10px;
          padding: 3rem 2rem;
          text-align: center;
          color: #484f58;
        ">
          <div style="font-size:2.5rem; margin-bottom:0.8rem;">🎯</div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
            Select your skills and click<br><b style="color:#8b949e;">Find My Tech Stack</b>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
#  EXPLAINER SECTION
# ──────────────────────────────────────────────────────────

st.markdown("---")

with st.expander("🔬 How the Algorithm Works (Theory)"):
    theory_col1, theory_col2 = st.columns(2)

    with theory_col1:
        st.markdown("""
**TF-IDF Vectorization**

Converts skill tags into numerical weights:

- **TF** (Term Frequency) = how often a skill appears within one job role's profile
- **IDF** (Inverse Document Frequency) = `log(total roles / roles with that skill)`

Specific skills like `Kubernetes` appear in few roles → **high IDF weight**.  
Generic skills like `Python` appear everywhere → **lower IDF weight**.

This prevents dominant generic terms from drowning out precise matches.
        """)

    with theory_col2:
        st.markdown("""
**Cosine Similarity**

Measures the **angle** between two vectors — not their magnitude.

```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
```

- **Score 1.0** → Vectors point in the same direction (perfect match)
- **Score 0.0** → Orthogonal (no shared characteristics)

Why not Euclidean distance?  
A role with 20 skills and one with 5 might be equally relevant — cosine handles this by ignoring vector length entirely.
        """)

    st.markdown("""
**The 4-Step Pipeline**

| Step | Name | What happens |
|------|------|-------------|
| 1 | **Ingestion** | User skills are cleaned, lowercased, and joined into a single string profile |
| 2 | **Scoring** | TF-IDF transforms user profile using the *fitted* vocabulary; cosine similarity computed vs. all roles |
| 3 | **Sorting** | All roles ranked by similarity score descending |
| 4 | **Filtering** | Top-N slice returned — prevents choice overload |

**Cold Start Handling**: If no skills are provided or all scores = 0, the system falls back to recommending the most skill-rich roles (popularity-based fallback), mirroring real production strategy.
    """)