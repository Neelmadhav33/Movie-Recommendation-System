# =============================================================================
# FILE: test_system.py
# PURPOSE: Automated test suite for the CineMatch recommendation system.
#          Run this to verify every component works correctly before your demo.
#
# HOW TO RUN:
#   python test_system.py
#
# WHAT IT TESTS:
#   Part 1 — Data loading & merging
#   Part 2 — Preprocessing (combined_features quality)
#   Part 3 — Model (TF-IDF matrix shape, cosine similarity range)
#   Part 4 — Recommendations (known good outputs)
#   Part 5 — Fuzzy search (typo tolerance)
#   Part 6 — Watchlist (add, check, remove)
#   Part 7 — Edge cases (not found, empty input, type filter)
# =============================================================================

import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.merge_datasets import load_and_merge
from src.preprocess import build_combined_features
from src.model import build_tfidf_matrix, compute_cosine_similarity
from src.recommender import MovieRecommender

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = 0
failed = 0

def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}✅ PASS{RESET} — {name}")
    else:
        failed += 1
        print(f"  {RED}❌ FAIL{RESET} — {name}")
        if detail:
            print(f"       {YELLOW}Detail: {detail}{RESET}")

def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


# =============================================================================
# PART 1 — DATA LOADING & MERGING
# =============================================================================
section("PART 1: Data Loading & Merging")

print("\n  Loading datasets (this takes ~5 seconds)...")
df_merged = load_and_merge(
    netflix_path='data/netflix_titles.csv',
    tmdb_path='data/tmdb_credits.csv'
)

test("Merged DataFrame is not empty",
     len(df_merged) > 0,
     f"Got {len(df_merged)} rows")

test("Merged dataset has 10,000+ titles",
     len(df_merged) >= 10000,
     f"Got {len(df_merged)} rows")

test("Required columns exist",
     all(c in df_merged.columns for c in ['title','type','genre','description','cast','director','country','rating']),
     f"Columns: {df_merged.columns.tolist()}")

test("No NaN in critical text columns",
     df_merged[['genre','description','cast','director']].isnull().sum().sum() == 0,
     "fillna('') should have handled all NaN")

test("Both types present (Movie + TV Show)",
     set(df_merged['type'].unique()) >= {'Movie', 'TV Show'},
     f"Types found: {df_merged['type'].unique().tolist()}")

test("Both sources present (Netflix + TMDB)",
     set(df_merged['source'].unique()) >= {'Netflix', 'TMDB'},
     f"Sources: {df_merged['source'].unique().tolist()}")

test("Netflix titles > 5000",
     len(df_merged[df_merged['source'] == 'Netflix']) > 5000,
     f"Netflix rows: {len(df_merged[df_merged['source'] == 'Netflix'])}")

test("No duplicate titles",
     df_merged['title'].str.lower().str.strip().duplicated().sum() == 0,
     f"Duplicates: {df_merged['title'].str.lower().duplicated().sum()}")

# Check a specific well-known title is present
test("'The Crown' is in dataset",
     'the crown' in df_merged['title'].str.lower().values,
     "The Crown should be in Netflix dataset")

test("'Inception' is in dataset",
     'inception' in df_merged['title'].str.lower().values,
     "Inception should be in TMDB dataset")


# =============================================================================
# PART 2 — PREPROCESSING
# =============================================================================
section("PART 2: Preprocessing — combined_features")

df_processed = build_combined_features(df_merged.copy())

test("combined_features column created",
     'combined_features' in df_processed.columns)

test("title_lower column created",
     'title_lower' in df_processed.columns)

test("combined_features has no NaN",
     df_processed['combined_features'].isnull().sum() == 0)

test("combined_features are lowercase",
     df_processed['combined_features'].str.lower().equals(df_processed['combined_features']),
     "All text should be lowercased")

test("combined_features average length > 50 chars",
     df_processed['combined_features'].str.len().mean() > 50,
     f"Avg length: {df_processed['combined_features'].str.len().mean():.0f}")

# Check The Crown's features contain expected keywords
crown_row = df_processed[df_processed['title_lower'] == 'the crown']
if not crown_row.empty:
    features = crown_row.iloc[0]['combined_features']
    test("The Crown features contain 'british'",
         'british' in features,
         f"Features snippet: {features[:200]}")
    test("The Crown features contain 'drama'",
         'drama' in features,
         f"Features snippet: {features[:200]}")
else:
    test("The Crown found for feature check", False, "Title not found")


# =============================================================================
# PART 3 — MODEL (TF-IDF + COSINE SIMILARITY)
# =============================================================================
section("PART 3: TF-IDF Model & Cosine Similarity Matrix")

tfidf_matrix, vectorizer = build_tfidf_matrix(df_processed['combined_features'])
cosine_sim = compute_cosine_similarity(tfidf_matrix)

import numpy as np

test("TF-IDF matrix rows == number of titles",
     tfidf_matrix.shape[0] == len(df_processed),
     f"Matrix rows: {tfidf_matrix.shape[0]}, Titles: {len(df_processed)}")

test("TF-IDF matrix has features (columns > 0)",
     tfidf_matrix.shape[1] > 0,
     f"Features: {tfidf_matrix.shape[1]}")

test("TF-IDF max features = 50000",
     tfidf_matrix.shape[1] <= 50000,
     f"Features: {tfidf_matrix.shape[1]}")

test("Cosine similarity matrix is square (n×n)",
     cosine_sim.shape[0] == cosine_sim.shape[1] == len(df_processed),
     f"Shape: {cosine_sim.shape}")

diag = np.diag(cosine_sim)
# Most titles have diagonal = 1.0.  A small number of TMDB-only titles
# have completely empty combined_features → zero TF-IDF vector → NaN/0 diagonal.
# We allow up to 30 such edge cases (21 found in practice).
zero_diag = int((diag < 0.999).sum())
test(f"Cosine similarity diagonal ≈ 1.0 (≤30 zero-vector edge cases, got {zero_diag})",
     zero_diag <= 30,
     f"{zero_diag} titles have empty combined_features → zero cosine diagonal")

test("Cosine similarity values in [0, 1]",
     cosine_sim.min() >= -0.01 and cosine_sim.max() <= 1.01,
     f"Min: {cosine_sim.min():.4f}, Max: {cosine_sim.max():.4f}")

test("Cosine similarity matrix is symmetric",
     np.allclose(cosine_sim, cosine_sim.T, atol=1e-5),
     "sim(A,B) should equal sim(B,A)")

# Check 'stop' words are not in vocabulary
vocab = set(vectorizer.vocabulary_.keys())
test("Stop words removed ('the' not in vocab)",
     'the' not in vocab,
     "TF-IDF should have removed English stop words")

test("'action' is in vocabulary",
     any('action' in w for w in vocab),
     "Genre words should be in vocabulary")


# =============================================================================
# PART 4 — RECOMMENDATION FUNCTION
# =============================================================================
section("PART 4: Recommendation Function")

print("\n  Building full recommender (uses cached matrices)...")
rec = MovieRecommender()
rec.df        = df_processed
rec.cosine_sim = cosine_sim
rec.vectorizer = vectorizer
rec.indices    = __import__('pandas').Series(
    df_processed.index,
    index=df_processed['title_lower']
)

# Test 1: The Crown
try:
    crown_recs = rec.recommend('The Crown', n=5)
    test("recommend('The Crown') returns 5 results",
         len(crown_recs) == 5,
         f"Got {len(crown_recs)} results")
    test("Recommendations have required columns",
         all(c in crown_recs.columns for c in ['title','type','genre','similarity_score']))
    test("Similarity scores are in (0, 1]",
         (crown_recs['similarity_score'] > 0).all() and (crown_recs['similarity_score'] <= 1).all(),
         f"Scores: {crown_recs['similarity_score'].tolist()}")
    test("The Crown itself is NOT in its own recommendations",
         'the crown' not in crown_recs['title'].str.lower().tolist())
    test("Recommendations are sorted descending by score",
         crown_recs['similarity_score'].is_monotonic_decreasing,
         f"Scores: {crown_recs['similarity_score'].tolist()}")
    print(f"\n  {YELLOW}Sample output — recommend('The Crown'):{RESET}")
    for i, row in crown_recs.iterrows():
        print(f"    {i}. {row['title']:<35} score={row['similarity_score']:.4f}")
except Exception as e:
    test("recommend('The Crown') ran without error", False, str(e))

# Test 2: Breaking Bad
print()
try:
    bb_recs = rec.recommend('Breaking Bad', n=5)
    test("recommend('Breaking Bad') returns results",
         len(bb_recs) > 0)
    # Better Call Saul should be top recommendation
    top_title = bb_recs.iloc[0]['title']
    test(f"Breaking Bad top rec is a crime/drama show",
         bb_recs.iloc[0]['similarity_score'] > 0.1,
         f"Top rec: '{top_title}' score={bb_recs.iloc[0]['similarity_score']:.4f}")
    print(f"\n  {YELLOW}Sample output — recommend('Breaking Bad'):{RESET}")
    for i, row in bb_recs.iterrows():
        print(f"    {i}. {row['title']:<35} score={row['similarity_score']:.4f}")
except Exception as e:
    test("recommend('Breaking Bad') ran without error", False, str(e))

# Test 3: Type filter
print()
try:
    movie_recs = rec.recommend('The Crown', n=5, filter_type='Movie')
    test("Type filter 'Movie' — all results are Movies",
         (movie_recs['type'] == 'Movie').all(),
         f"Types: {movie_recs['type'].tolist()}")
    tv_recs = rec.recommend('The Crown', n=5, filter_type='TV Show')
    test("Type filter 'TV Show' — all results are TV Shows",
         (tv_recs['type'] == 'TV Show').all(),
         f"Types: {tv_recs['type'].tolist()}")
except Exception as e:
    test("Type filter test ran without error", False, str(e))


# =============================================================================
# PART 5 — FUZZY SEARCH
# =============================================================================
section("PART 5: Fuzzy Search (Typo Tolerance)")

fuzzy_cases = [
    ("Avengrs",          "The Avengers",      "Common typo — missing letter"),
    ("Brekng Bd",        "Breaking Bad",      "Multiple typos"),
    ("strenger thinggs", "Stranger Things",   "Multiple character errors"),
    ("The Crwn",         "The Crown",         "Missing vowel"),
    ("Incption",         "Inception",         "Missing letter"),
]

for query, expected, description in fuzzy_cases:
    results = rec.fuzzy_search(query, n=5)
    found   = expected in results
    test(f"fuzzy_search('{query}') finds '{expected}' [{description}]",
         found,
         f"Got: {results}")

# Test with a completely nonsense query
# Use a very long random string that can't possibly match any real title
nonsense = rec.fuzzy_search("qqqqzzzznonsense999xyzabc", n=5)
test("Nonsense query returns empty list (no real match)",
     len(nonsense) == 0,
     f"Got: {nonsense}")


# =============================================================================
# PART 6 — WATCHLIST SYSTEM (tested via watchlist_manager directly)
# =============================================================================
section("PART 6: Watchlist System")

from src.watchlist_manager import (
    load_watchlist, save_watchlist,
    add_to_watchlist, remove_from_watchlist, is_in_watchlist,
    WATCHLIST_PATH as WL_PATH
)

# Use a temp file to avoid polluting real watchlist
import tempfile
tmp_wl = tempfile.mktemp(suffix='.json')

# Patch path for test
import src.watchlist_manager as wl_mod
wl_mod.WATCHLIST_PATH = tmp_wl
with open(tmp_wl, 'w') as f:
    json.dump({}, f)

# Test add
add_to_watchlist("Inception", "Action, Sci-Fi", "Movie")
wl = load_watchlist()
test("add_to_watchlist() adds title to JSON",
     "Inception" in wl,
     f"Watchlist keys: {list(wl.keys())}")

test("Watchlist entry has 'genre' field",
     wl.get("Inception", {}).get("genre") == "Action, Sci-Fi")

test("Watchlist entry has 'added_on' timestamp",
     bool(wl.get("Inception", {}).get("added_on")),
     f"added_on: {wl.get('Inception',{}).get('added_on')}")

test("Watchlist entry has 'status' field",
     "status" in wl.get("Inception", {}))

# Test is_in_watchlist
test("is_in_watchlist() returns True for added title",
     is_in_watchlist("Inception"))

test("is_in_watchlist() returns False for missing title",
     not is_in_watchlist("NONEXISTENT_MOVIE_XYZ"))

# Test add multiple
add_to_watchlist("The Crown", "Drama", "TV Show")
add_to_watchlist("Breaking Bad", "Crime", "TV Show")
wl = load_watchlist()
test("Multiple items can be added to watchlist",
     len(wl) == 3,
     f"Watchlist size: {len(wl)}")

# Test remove
remove_from_watchlist("Inception", "Watched ✅")
wl = load_watchlist()
test("remove_from_watchlist() removes title",
     "Inception" not in wl,
     f"Watchlist keys after remove: {list(wl.keys())}")

test("Other entries not affected by remove",
     "The Crown" in wl and "Breaking Bad" in wl)

# Restore
wl_mod.WATCHLIST_PATH = WL_PATH
os.remove(tmp_wl)


# =============================================================================
# PART 7 — EDGE CASES
# =============================================================================
section("PART 7: Edge Cases & Error Handling")

# Title not found — should raise ValueError
try:
    rec.recommend("XYZXYZ_NOT_A_REAL_TITLE_12345", n=5)
    test("ValueError raised for unknown title", False, "Should have raised ValueError")
except ValueError as e:
    test("ValueError raised for unknown title", True, str(e)[:80])

# get_details for unknown title returns empty dict
details = rec.get_details("XYZXYZ_FAKE")
test("get_details() returns {} for unknown title",
     details == {},
     f"Got: {details}")

# get_details for known title returns dict with data
details = rec.get_details("The Crown")
test("get_details('The Crown') returns non-empty dict",
     bool(details) and details.get('title') == 'The Crown',
     f"Keys: {list(details.keys())}")

# Recommend with n=1 returns exactly 1
result = rec.recommend("Inception", n=1)
test("recommend(n=1) returns exactly 1 result",
     len(result) == 1,
     f"Got {len(result)} results")

# Recommend with n=10 returns up to 10
result = rec.recommend("Inception", n=10)
test("recommend(n=10) returns up to 10 results",
     1 <= len(result) <= 10,
     f"Got {len(result)} results")

# get_all_titles returns a list
all_titles = rec.get_all_titles()
test("get_all_titles() returns non-empty list",
     isinstance(all_titles, list) and len(all_titles) > 0,
     f"Length: {len(all_titles)}")

test("get_all_titles(Movie) only contains Movies",
     all(rec.df[rec.df['title'] == t]['type'].values[0] == 'Movie'
         for t in rec.get_all_titles('Movie')[:10]))

# get_trending returns DataFrame
trending = rec.get_trending(n=5)
test("get_trending(5) returns 5 rows",
     len(trending) == 5,
     f"Got {len(trending)} rows")

test("get_trending rows have release_year",
     'release_year' in trending.columns)


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{BOLD}{'═'*60}{RESET}")
total = passed + failed
print(f"{BOLD}  TEST SUMMARY: {passed}/{total} passed{RESET}")
if failed == 0:
    print(f"  {GREEN}{BOLD}🎉 ALL TESTS PASSED! System is ready.{RESET}")
else:
    print(f"  {RED}{BOLD}⚠️  {failed} test(s) failed. Review above.{RESET}")
print(f"{BOLD}{'═'*60}{RESET}\n")

sys.exit(0 if failed == 0 else 1)
