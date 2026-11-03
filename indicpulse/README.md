# IndicPulse 🇮🇳
### Cross-Platform Indian Semantic & Sentiment Analyzer

> A 5-step AI analytics engine that ingests comments from any digital source, decodes Romanized code-mixed Hinglish, accounts for emojis and sarcasm, filters toxic abuse, and delivers engagement-weighted insights via a SaaS-grade Streamlit dashboard.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd indicpulse
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Select **Demo Dataset** to try immediately — no API keys needed.

---

## 🏗️ Architecture

```
indicpulse/
├── app.py                          ← Streamlit frontend entry point
├── requirements.txt
├── .env.example
├── core/
│   ├── ingestion.py                ← Step 1: YouTube API + CSV Parser
│   ├── preprocessor.py             ← Step 2: Emoji tags, profanity, Hinglish
│   ├── mood_classifier.py          ← Step 3: MuRIL 4-class classifier
│   ├── topic_clusterer.py          ← Step 4: IndicSBERT + BERTopic
│   ├── summarizer.py               ← Step 5: Gemini 2.5 Flash agent
│   ├── pipeline.py                 ← Master orchestrator
│   └── utils.py
├── ui/
│   ├── styles.py                   ← Glassmorphic SaaS CSS
│   └── charts.py                   ← Plotly chart components
├── models/
│   └── muril_finetuned/            ← Fine-tuned weights (after Colab training)
├── data/
│   ├── sample_reviews.csv          ← Demo dataset
│   └── training_labels.csv         ← MuRIL training data
└── notebooks/
    ├── 01_finetune_muril_colab.ipynb   ← Colab fine-tuning guide
    └── 02_explore_pipeline.ipynb       ← Interactive exploration
```

---

## 🧠 The 5-Step Pipeline

| Step | Component | Technology |
|------|-----------|------------|
| **1. Ingest** | Universal Ingestion Engine | YouTube Data API v3 + CSV Parser |
| **2. Preprocess** | Smart Text Cleaner | `re`, `emoji`, custom rules |
| **3. Classify** | Mood & Toxicity | `google/muril-base-cased` (fine-tuned) |
| **4. Cluster** | Semantic Topics | IndicSBERT + BERTopic (UMAP + HDBSCAN) |
| **5. Summarize** | AI Executive Report | Gemini 2.5 Flash |

### Step 2 – Preprocessing Details

- **Emoji → Semantic Tags:** `🔥` → `[fire_hype]`, `🤡` → `[clown_mocking]` (for MuRIL)
- **Emoji Strip:** Clean text for BERTopic (topics = subjects, not emoji types)
- **Elongation:** `baaaahut` → `baahut`
- **Profanity:** Un-censor (`ch*tiya` → `chutiya`) + flag `has_profanity = True`
- **Slang:** `btw` → `by the way`, `fr` → `for real`

### Step 3 – Label Schema

| ID | Label | Description | Color |
|----|-------|-------------|-------|
| 0 | **Positive** | Genuine praise, hype | 🟢 Emerald |
| 1 | **Sarcastic** | Trolling, irony | 🟡 Amber |
| 2 | **Neutral** | Informational, mixed | ⬜ Gray |
| 3 | **Toxic** | Abusive, profane | 🔴 Coral Red |

### Step 5 – Engagement Weight Formula

$$W = \log_{10}(\text{Likes} + 1) + 1$$

Never zero, log-scaled, ensures high-liked comments rank higher in cluster summaries.

---

## 💻 Usage Modes

### Demo Mode (No API Keys)
```python
from core.pipeline import run_pipeline

result = run_pipeline(source="demo")
print(result.mood_counts)
print(result.pulse_score)
```

### YouTube Mode
```python
result = run_pipeline(
    source="youtube",
    video_url="https://youtu.be/VIDEO_ID",
    api_key_youtube="YOUR_YT_KEY",
    api_key_gemini="YOUR_GEMINI_KEY",
    max_comments=500,
)
```

### CSV Mode
```python
result = run_pipeline(
    source="csv",
    csv_path="data/my_reviews.csv",
    api_key_gemini="YOUR_GEMINI_KEY",
)
```

---

## 🏋️ Fine-tuning MuRIL (Google Colab)

1. Open `notebooks/01_finetune_muril_colab.ipynb` in Google Colab
2. Select **T4 GPU** runtime (free tier)
3. Upload `data/training_labels.csv`
4. Run all cells (~20-30 min)
5. Download `muril_finetuned.zip` and extract to `models/muril_finetuned/`

**Without fine-tuned weights:** The pipeline automatically falls back to rule-based heuristics, which work well for obvious patterns.

---

## 🔑 API Keys

| Key | Required | Get it at |
|-----|----------|-----------|
| `YOUTUBE_API_KEY` | For YouTube ingestion | [console.cloud.google.com](https://console.cloud.google.com/apis/credentials) |
| `GEMINI_API_KEY` | For AI executive summary | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

Both are **free tier** compatible.

---

## 🎨 UI Features

- **Glassmorphic metric cards** with hover glow effects
- **Mood donut chart** with Emerald/Amber/Gray/Red color coding
- **Pulse gauge** (1–10 score from Gemini)
- **Sentiment timeline** (mood over time)
- **Topic engagement bar chart** (log-scaled)
- **Topic bubble map** (size vs. engagement scatter)
- **Toxic quarantine box** with severity filter + CSV export
- **4-part AI report** (Loved / Complaints / Action Plan / Verdict)

---

## 💡 Tech Stack (100% Free & Open-Source)

| Layer | Technology | Cost |
|-------|-----------|------|
| Backend | Python 3.10+, Pandas, NumPy | Free |
| Ingestion | YouTube Data API v3 | Free (10K queries/day) |
| Preprocessing | `re`, `emoji` | Free |
| Mood AI | `google/muril-base-cased` (HuggingFace) | Free |
| GPU Training | Google Colab T4 | Free |
| Topic AI | IndicSBERT + BERTopic | Free |
| Summary AI | Gemini 2.5 Flash | Free (1M tokens/day) |
| Frontend | Streamlit + Plotly | Free |

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Analytics Dashboard** | Donut chart, pulse gauge, timeline, engagement bars |
| **Semantic Explorer** | Topic bubble map, cluster deep-dives with keywords |
| **Toxic Moderation** | Quarantine queue, severity filter, export |
| **AI Executive Report** | 4-part Gemini-generated creator action plan |

---

*Built for Indian creators, brands, and researchers who need to understand what their Hindi/Hinglish-speaking audience actually thinks.*
