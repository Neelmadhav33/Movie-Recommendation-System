# 🔮 Future Improvements
## CineMatch — Roadmap for a Production-Grade System

> Each improvement is explained with WHY it helps, HOW to implement it,
> and what libraries/code to use. Perfect for explaining in your viva.

---

## 1. 🤝 Collaborative Filtering (User Ratings Dataset)

### Why?
Our current system is **content-based only** — it recommends movies similar
to a selected title based on features. It does NOT know:
- What YOU personally like
- That two stylistically different movies are loved by the same audience

Collaborative filtering fixes this by learning from user behavior patterns.

### What to Add
- Source the **MovieLens 25M dataset** (25 million ratings, 62,000 movies)
  - Download: https://grouplens.org/datasets/movielens/25m/
- Build a **user-item rating matrix**: rows = users, columns = movies, values = ratings (1–5)

### Implementation (Item-Based Collaborative Filtering)

```python
# src/collaborative.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_item_similarity(ratings_df: pd.DataFrame):
    """
    Build item-item cosine similarity from a user-rating matrix.

    Parameters
    ----------
    ratings_df : DataFrame with columns [userId, movieId, rating]

    Returns
    -------
    item_sim : DataFrame (movieId × movieId) cosine similarity
    """
    # Pivot to user-item matrix
    matrix = ratings_df.pivot_table(
        index='userId', columns='movieId', values='rating'
    ).fillna(0)

    # Cosine similarity between items (columns)
    item_sim = cosine_similarity(matrix.T)
    return pd.DataFrame(item_sim, index=matrix.columns, columns=matrix.columns)

def cf_recommend(movie_id: int, item_sim: pd.DataFrame, n: int = 5) -> list:
    """Return top-n most similar movies to movie_id."""
    if movie_id not in item_sim.index:
        return []
    scores = item_sim[movie_id].sort_values(ascending=False)
    return scores.iloc[1:n+1].index.tolist()  # skip itself
```

### Install
```bash
pip install scikit-surprise  # for matrix factorization (SVD)
```

---

## 2. 🔀 Hybrid Recommendation System

### Why?
Neither content-based nor collaborative filtering alone is ideal:

| Problem | Content-Based | Collaborative |
|---------|:---:|:---:|
| New user cold start | ✅ | ❌ |
| New item cold start | ✅ | ❌ |
| Personalization | ❌ | ✅ |
| Serendipitous picks | ❌ | ✅ |

A **hybrid system** combines both to get the best of each.

### Two Hybrid Approaches

#### Approach A — Weighted Score Fusion
```python
def hybrid_recommend(title: str, user_id: int, alpha: float = 0.6, n: int = 5):
    """
    Combine content-based and collaborative scores.

    alpha = weight for content-based (0.0 to 1.0)
    1 - alpha = weight for collaborative
    """
    content_scores = content_recommender.get_scores(title)    # your existing system
    collab_scores  = cf_recommender.get_scores(user_id)       # from collaborative filter

    # Normalize both to [0, 1]
    content_norm = (content_scores - content_scores.min()) / (content_scores.max() - content_scores.min())
    collab_norm  = (collab_scores  - collab_scores.min())  / (collab_scores.max()  - collab_scores.min())

    # Weighted combination
    hybrid_scores = alpha * content_norm + (1 - alpha) * collab_norm
    return hybrid_scores.sort_values(ascending=False).head(n)
```

#### Approach B — Switching Hybrid
```python
def switching_recommend(title: str, user_id: int, min_ratings: int = 10):
    """
    Use content-based for new users, collaborative for returning users.
    """
    user_rating_count = ratings_df[ratings_df['userId'] == user_id].shape[0]

    if user_rating_count < min_ratings:
        # New user → content-based (no cold start problem)
        return content_recommender.recommend(title)
    else:
        # Returning user → collaborative (personalized)
        return cf_recommender.recommend(user_id)
```

---

## 3. 🧠 Neural Collaborative Filtering (Deep Learning)

### Why?
Matrix factorization (SVD) learns **linear** patterns.
Neural networks learn **non-linear** patterns — much richer representations.

### Architecture (Neural Matrix Factorization)

```
User ID ──▶ Embedding(64) ──┐
                             ├──▶ Dense(128) ──▶ Dense(64) ──▶ Dense(1) ──▶ Predicted Rating
Item ID ──▶ Embedding(64) ──┘
```

```python
# src/neural_cf.py
import tensorflow as tf

def build_ncf_model(n_users: int, n_items: int, embedding_dim: int = 64):
    """
    Neural Collaborative Filtering model.

    Parameters
    ----------
    n_users       : int   Number of unique users
    n_items       : int   Number of unique items (movies)
    embedding_dim : int   Latent factor dimension
    """
    # User embedding — maps each user ID to a dense vector
    user_input    = tf.keras.Input(shape=(1,), name='user')
    user_embed    = tf.keras.layers.Embedding(n_users, embedding_dim)(user_input)
    user_flat     = tf.keras.layers.Flatten()(user_embed)

    # Item embedding — maps each movie ID to a dense vector
    item_input    = tf.keras.Input(shape=(1,), name='item')
    item_embed    = tf.keras.layers.Embedding(n_items, embedding_dim)(item_input)
    item_flat     = tf.keras.layers.Flatten()(item_embed)

    # Concatenate user + item embeddings
    concat        = tf.keras.layers.Concatenate()([user_flat, item_flat])

    # Fully connected layers learn complex interaction patterns
    x = tf.keras.layers.Dense(128, activation='relu')(concat)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64,  activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)   # predicted rating

    model = tf.keras.Model(inputs=[user_input, item_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train:
# model = build_ncf_model(n_users=10000, n_items=10164)
# model.fit([user_ids, item_ids], ratings, epochs=10, batch_size=256)
```

### Install
```bash
pip install tensorflow
```

---

## 4. 🎬 Movie Poster Images (TMDB API)

### Why?
A visual UI with actual movie posters looks dramatically more professional.
TMDB has a free public API that returns poster image URLs.

### Implementation
```python
import requests

TMDB_API_KEY = "YOUR_API_KEY"  # Get free at https://www.themoviedb.org/settings/api
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

def get_poster_url(movie_title: str) -> str:
    """Fetch the poster URL for a movie title from TMDB API."""
    url = f"https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": movie_title}
    response = requests.get(url, params=params)
    results = response.json().get("results", [])
    if results and results[0].get("poster_path"):
        return TMDB_IMG_BASE + results[0]["poster_path"]
    return None

# In Streamlit:
# poster_url = get_poster_url("Inception")
# if poster_url:
#     st.image(poster_url, width=200)
```

---

## 5. 👤 User Login & Personalised Profiles

### Why?
Currently, the watchlist is shared for all users of the same machine.
Real systems have per-user profiles so every user gets personalised recommendations.

### Simple Implementation with Streamlit + SQLite
```python
# src/user_manager.py
import sqlite3, hashlib

def init_db():
    conn = sqlite3.connect('data/users.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE,
            password  TEXT,
            created   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            user_id   INTEGER,
            title     TEXT,
            genre     TEXT,
            type      TEXT,
            added_on  TEXT,
            status    TEXT,
            PRIMARY KEY (user_id, title)
        )
    """)
    conn.commit()
    conn.close()

def register_user(username: str, password: str) -> bool:
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn = sqlite3.connect('data/users.db')
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False  # Username taken
```

---

## 6. 📊 Personalised "Because You Watched" Section

### Why?
Netflix's most powerful UX feature — showing users recommendations tied to
specific titles they watched, not just a generic list.

### Implementation
```python
def because_you_watched(watchlist: dict, recommender, n_per_title: int = 3) -> dict:
    """
    For each watched title in watchlist, get n recommendations.

    Returns
    -------
    dict: { "Because you watched X": [rec1, rec2, rec3], ... }
    """
    result = {}
    watched = [t for t, info in watchlist.items() if info.get('status') == 'Watched']
    for title in watched[:5]:  # limit to 5 source titles
        try:
            recs = recommender.recommend(title, n=n_per_title)
            result[f"Because you watched: {title}"] = recs
        except ValueError:
            pass
    return result
```

---

## 7. ⚡ Performance Optimisation (Caching the Model)

### Why?
The TF-IDF + cosine similarity computation takes ~15 seconds on first load.
Saving the matrix to disk means subsequent loads take < 1 second.

### Implementation
```python
# src/cache_manager.py
import pickle
import os

CACHE_DIR = "cache/"

def save_model_cache(cosine_sim, vectorizer, df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(f"{CACHE_DIR}cosine_sim.pkl",  "wb") as f: pickle.dump(cosine_sim,  f)
    with open(f"{CACHE_DIR}vectorizer.pkl",  "wb") as f: pickle.dump(vectorizer,  f)
    with open(f"{CACHE_DIR}dataframe.pkl",   "wb") as f: pickle.dump(df,          f)
    print("✅ Model cache saved.")

def load_model_cache():
    try:
        with open(f"{CACHE_DIR}cosine_sim.pkl",  "rb") as f: cosine_sim  = pickle.load(f)
        with open(f"{CACHE_DIR}vectorizer.pkl",  "rb") as f: vectorizer  = pickle.load(f)
        with open(f"{CACHE_DIR}dataframe.pkl",   "rb") as f: df          = pickle.load(f)
        print("✅ Model loaded from cache (fast).")
        return cosine_sim, vectorizer, df
    except FileNotFoundError:
        return None, None, None

# Usage in recommender.py fit():
# cosine_sim, vectorizer, df = load_model_cache()
# if cosine_sim is None:
#     # recompute
#     cosine_sim = compute_cosine_similarity(tfidf_matrix)
#     save_model_cache(cosine_sim, vectorizer, df)
```

---

## 8. 📈 Analytics Dashboard

### Why?
Shows how the recommendation system is performing — useful for
demonstrating the project at a viva or hackathon.

### Metrics to Track
- Most recommended titles
- Most added to watchlist
- Average similarity score per recommendation
- Genre diversity in recommendations

```python
# In Streamlit analytics tab:
st.subheader("📊 System Analytics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Titles", f"{len(rec.df):,}")
col2.metric("Watchlist Items", len(load_watchlist()))
col3.metric("TF-IDF Features", "50,000")
```

---

## 📅 Implementation Priority Order

| Priority | Feature | Difficulty | Time Estimate |
|:---:|--------|:---:|:---:|
| 🥇 1 | Poster images (TMDB API) | Easy | 2 hours |
| 🥇 2 | Model caching with pickle | Easy | 1 hour |
| 🥇 3 | Because You Watched section | Easy | 2 hours |
| 🥈 4 | User login with SQLite | Medium | 1 day |
| 🥈 5 | Collaborative filtering (MovieLens) | Medium | 2 days |
| 🥈 6 | Weighted hybrid system | Medium | 1 day |
| 🥉 7 | Neural Collaborative Filtering | Hard | 1 week |
| 🥉 8 | Full production deployment | Hard | 2 weeks |
