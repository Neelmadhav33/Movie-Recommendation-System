[README.md](https://github.com/user-attachments/files/26887479/README.md)
# 🎬 CineMatch — Movie Discovery & Recommendation System

> An IMDb-inspired movie discovery platform powered by **TF-IDF + Cosine Similarity**, built with Python and Streamlit. Merges Netflix and TMDB datasets into one unified 10,000+ title catalog.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Fuzzy Search** | Type "Avengrs" → suggests "The Avengers" (typo-tolerant via difflib) |
| 🍿 **Recommendations** | Top 5 "More Like This" using TF-IDF + Cosine Similarity |
| 📋 **Watchlist** | Add/Remove with reason (Watched ✅, Changed mind ❌, Other 📝) |
| 🔥 **Trending** | Most recently released titles |
| ⭐ **Top Rated** | Titles with established content ratings |
| 🆕 **Recently Added** | Latest Netflix additions |
| 🎭 **Type Filter** | Filter results by Movie or TV Show |
| 🌍 **10,164 Titles** | Netflix (5,837) + TMDB (4,803) merged catalog |

---

## 📁 Project Structure

```
movie_discovery/
│
├── data/
│   ├── netflix_titles.csv          ← Netflix dataset (5,837 titles)
│   └── tmdb_credits.csv            ← TMDB credits dataset (4,803 movies)
│
├── src/
│   ├── merge_datasets.py           ← Load, parse, unify, merge both datasets
│   ├── preprocess.py               ← Build combined_features column, clean text
│   ├── model.py                    ← TF-IDF vectorization + cosine similarity
│   └── recommender.py              ← Main engine: recommend(), fuzzy_search(), etc.
│
├── app/
│   └── app.py                      ← Netflix-style Streamlit UI
│
├── watchlist/
│   └── watchlist.json              ← Persistent watchlist storage
│
├── requirements.txt
└── README.md
```

**File purposes at a glance:**

| File | Job |
|------|-----|
| `merge_datasets.py` | Parses TMDB JSON, renames columns to common schema, left-joins on title |
| `preprocess.py` | Combines genre+description+cast+director+country → `combined_features` |
| `model.py` | TF-IDF (50k features, bigrams) → 10164×10164 cosine similarity matrix |
| `recommender.py` | `MovieRecommender` class — fit(), recommend(), fuzzy_search(), trending |
| `app/app.py` | Full Streamlit UI with dark theme, watchlist, sidebars, tabs |
| `watchlist.json` | Persistent JSON storage for the user's saved titles |

---

## 📊 Datasets

### Dataset 1 — Netflix Titles (Nov 2019)
- **Size:** 5,837 titles (Movies + TV Shows)
- **Source:** Kaggle — Netflix Movies and TV Shows
- **Key columns:** `title`, `type`, `director`, `cast`, `country`, `release_year`, `rating`, `duration`, `listed_in`, `description`

### Dataset 2 — TMDB 5000 Credits
- **Size:** 4,803 movies
- **Source:** Kaggle — TMDB Movie Metadata
- **Key columns:** `title`, `cast` (JSON), `crew` (JSON with Director info)
- **Note:** Cast and crew are stored as JSON arrays and parsed to plain strings.

### After Merging
- **Total:** 10,164 unique titles
- **Netflix titles enriched with TMDB cast:** 424
- **TMDB-only movies added:** 4,389
- **Duplicates removed:** 62

---

## ⚙️ How the System Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                           │
│                                                                 │
│  Netflix CSV ──┐                                                │
│                ├──▶ merge_datasets.py ──▶ Unified DataFrame     │
│  TMDB CSV ─────┘   (parse JSON, rename,   (10,164 titles)       │
│                     join on title)                              │
│                                   │                             │
│                                   ▼                             │
│                         preprocess.py                           │
│              genre + description + cast + director + country    │
│                      ──▶  combined_features  ──▶  clean text    │
│                                   │                             │
│                                   ▼                             │
│                            model.py                             │
│               TfidfVectorizer(50k features, bigrams)            │
│                      ──▶  TF-IDF Matrix (10164 × 50000)         │
│                      ──▶  Cosine Similarity (10164 × 10164)     │
│                                   │                             │
│                                   ▼                             │
│                          recommender.py                         │
│              recommend(title) → sort scores → top N results     │
│              fuzzy_search(query) → difflib close matches        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 How Recommendations Work (Step by Step)

1. User selects **"The Crown"**
2. System finds its row index in the DataFrame (e.g. index 42)
3. Looks up row 42 in the **10164×10164 cosine similarity matrix**
4. Gets similarity score between "The Crown" and every other title
5. Sorts scores descending: `[0.21, 0.19, 0.18, ...]`
6. Skips the first result (The Crown itself, score = 1.0)
7. Returns **top 5**: "Black Earth Rising", "Call the Midwife", etc.

---

## 🔍 Content-Based Filtering — Explanation

**What is it?**
Recommend items based on the **properties of the items themselves**, not on what other users liked.

**How it works here:**
- Each title is described by: genre, description, cast, director, country
- These are combined into one text string (`combined_features`)
- TF-IDF converts that text into a numerical vector
- Cosine similarity measures how close two vectors are
- Closest vectors = most similar titles → recommended

**Advantages:** Works without user data, no cold start problem, transparent
**Disadvantages:** Limited to selected features, cannot discover unexpected matches

---

## 👥 Collaborative Filtering — Theory

**What is it?**
Recommend items based on **what similar users liked**, not item properties.

> "Users who liked what you liked, also liked THIS."

**Types:**
- **User-based:** Find users with similar taste → recommend what they liked
- **Item-based:** Find items similar to ones you rated → recommend those
- **Matrix Factorization (SVD):** Decompose the ratings matrix into latent factors

**We don't implement this** because we have no user rating data. In a real system, both approaches are combined (hybrid).

---

## 🔬 Cold Start Problem

When the system **can't make good recommendations** due to lack of data:
- **New User:** No history → collaborative filter has nothing to learn from
- **New Item:** No ratings → collaborative filter can't recommend it

**Solution:** Use content-based filtering (like ours) as a fallback. It only needs item features — no user history required.

---

## 🏗️ Candidate Generation & Ranking (Real-World Systems)

**The Problem:** Netflix has 15,000+ titles. You can't score ALL of them for every request in real time.

**Solution: 2-Stage Pipeline**

```
All 15,000+ titles
      │
      ▼  [Stage 1: Candidate Generation — fast, broad]
  ~200 candidates  ← Our TF-IDF system does this stage!
      │
      ▼  [Stage 2: Ranking — slow, personalized]
  Top 10 shown to user ← Real Netflix uses neural networks here
```

- **Stage 1 (Candidate Generation):** Fast, lightweight filter. Picks ~200 relevant titles. Uses embedding similarity or our TF-IDF cosine similarity.
- **Stage 2 (Ranking):** Slower, uses richer signals: user history, device, time of day, predicted click probability. Often a deep learning model.

---

## 🛠️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/app.py
```

The app opens at `http://localhost:8501`

---

## ▶️ How to Run

### Streamlit App
```bash
streamlit run app/app.py
```

### Test individual modules
```bash
python src/merge_datasets.py    # Test data loading & merging
python src/preprocess.py        # Test feature engineering
python src/model.py             # Test TF-IDF + cosine similarity
python src/recommender.py       # Test recommendation engine
```

### Python REPL
```python
from src.recommender import MovieRecommender

rec = MovieRecommender()
rec.fit()

# Get recommendations
print(rec.recommend("The Crown", n=5))

# Fuzzy search (typo-tolerant)
print(rec.fuzzy_search("Avengrs"))

# Get movie details
print(rec.get_details("Inception"))
```

---

## 📸 Screenshots

```
┌─────────────────────────────────────────────────────────────────┐
│  🎬 CineMatch              🔍 Search movies, shows, actors...   │
├──────────────────────────────────┬──────────────────────────────┤
│  🎬 The Crown                    │  🔥 Trending                 │
│  [TV Show] [TV-MA] [2019]        │  · Chocolate (2020)          │
│                                  │  · Yankee (2019)             │
│  "This drama follows the         │                              │
│   political rivalries..."        │  ⭐ Popular                  │
│                                  │  · The Crown                 │
│  👤 Claire Foy                   │  · Breaking Bad              │
│  👤 John Lithgow                 │                              │
│                                  │  📊 DATASET                  │
│  [➕ Add to Watchlist]           │  🎬 8,294 Movies             │
│                                  │  📺 1,870 TV Shows           │
├──────────────────────────────────┴──────────────────────────────┤
│  🍿 More Like This                                              │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Black Earth Rising│  │ Call the Midwife  │                   │
│  │ British TV · 21% │  │ British TV · 19% │                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔮 Future Improvements

See `FUTURE_IMPROVEMENTS.md` for full details.

1. **Add user ratings** → enable collaborative filtering
2. **Hybrid system** → combine content-based + collaborative (weighted)
3. **Poster images** → TMDB API integration for movie artwork
4. **Neural CF** → learn user/item embeddings with deep learning
5. **Persistent user profiles** → login system with saved preferences

---

## 📚 Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` | Data manipulation, DataFrame operations |
| `scikit-learn` | TF-IDF vectorization, cosine similarity |
| `streamlit` | Web UI framework |
| `difflib` | Fuzzy string matching (Python built-in) |
| `json` | Parsing TMDB cast/crew, watchlist storage |

---

## 📄 License

Educational use only. Datasets sourced from Kaggle (Netflix + TMDB).
