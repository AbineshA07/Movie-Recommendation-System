import os
from pathlib import Path
import pickle
import requests
import pandas as pd
import streamlit as st

# ---- Configuration ----
# Ensure TMDB API key is set as an environment variable for security:
#   export TMDB_API_KEY="your_api_key_here"   (Linux/macOS)
#   setx TMDB_API_KEY "your_api_key_here"    (Windows - restart shell)
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

# ---- Helper: locate the pickle file (same folder as this script) ----
def get_default_data_path(filename="movie_data.pkl"):
    try:
        base = Path(__file__).parent
    except NameError:
        # __file__ may not exist in some interactive environments; fallback to cwd
        base = Path.cwd()
    return base / filename

DATA_PATH = get_default_data_path()

# ---- Load processed data and similarity matrix ----
@st.cache_data(show_spinner=False)
def load_data(pickle_path: Path):
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Could not find '{pickle_path.name}' in the repository. "
            "Place the file in the same folder as this script."
        )
    with open(pickle_path, "rb") as f:
        obj = pickle.load(f)
    # Accept either a tuple (movies, cosine_sim) or a dict with keys
    if isinstance(obj, tuple) and len(obj) == 2:
        movies_df, cosine_sim = obj
    elif isinstance(obj, dict) and "movies" in obj and "cosine_sim" in obj:
        movies_df = obj["movies"]
        cosine_sim = obj["cosine_sim"]
    else:
        # Try to infer common structure
        try:
            movies_df = obj[0]
            cosine_sim = obj[1]
        except Exception as e:
            raise ValueError(
                "The pickle file doesn't contain the expected (movies, cosine_sim) structure."
            ) from e

    if not isinstance(movies_df, pd.DataFrame):
        raise TypeError("Loaded 'movies' must be a pandas DataFrame.")
    return movies_df.reset_index(drop=True), cosine_sim

try:
    movies, cosine_sim = load_data(DATA_PATH)
except Exception as e:
    st.title("Movie Recommendation System")
    st.error(f"Error loading data: {e}")
    st.stop()

# Ensure required columns exist
required_columns = {"title", "movie_id"}
if not required_columns.issubset(movies.columns):
    st.title("Movie Recommendation System")
    st.error(
        f"The movies DataFrame must contain the columns: {required_columns}. "
        f"Found columns: {list(movies.columns)}"
    )
    st.stop()

# ---- Recommendation function ----
@st.cache_data(show_spinner=False)
def get_recommendations(title: str, movies_df: pd.DataFrame = movies, cosine_sim_matrix=None, top_n=10):
    if cosine_sim_matrix is None:
        raise ValueError("cosine_sim matrix is required.")
    # Find title index safely (case-sensitive match first, fallback to case-insensitive)
    matches = movies_df[movies_df['title'] == title]
    if matches.empty:
        # try case-insensitive
        matches = movies_df[movies_df['title'].str.lower() == title.lower()]
        if matches.empty:
            raise ValueError(f"Movie title '{title}' not found in dataset.")
    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Exclude the movie itself and take top_n
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]
    movie_indices = [i for i, score in sim_scores]
    return movies_df[['title', 'movie_id']].iloc[movie_indices].reset_index(drop=True)

# ---- TMDB poster fetch ----
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int):
    if TMDB_API_KEY is None:
        # Return placeholder or None if no key provided
        return None
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return None
    poster_path = data.get("poster_path")
    if not poster_path:
        return None
    return f"https://image.tmdb.org/t/p/w500{poster_path}"

# ---- Streamlit UI ----
st.set_page_config(page_title="Movie Recommendation System", layout="wide")
st.title("🎬 Movie Recommendation System")

st.write(
    "Select a movie and click **Recommend**. "
    "To show posters the repository must be configured with a TMDB API key in the environment variable `TMDB_API_KEY`."
)

# Sort and show unique titles
titles = movies['title'].dropna().unique()
titles_sorted = sorted(titles)
selected_movie = st.selectbox("Select a movie:", titles_sorted)

if st.button("Recommend"):
    if not selected_movie:
        st.error("Please select a movie.")
    else:
        try:
            recommendations = get_recommendations(selected_movie, movies, cosine_sim, top_n=10)
        except Exception as e:
            st.error(f"Could not compute recommendations: {e}")
        else:
            st.write("Top 10 recommended movies:")

            # Display grid -- 2 rows x 5 columns
            with st.container():
                num = len(recommendations)
                per_row = 5
                rows = (num + per_row - 1) // per_row
                idx = 0
                for r in range(rows):
                    cols = st.columns(per_row)
                    for c, col in zip(range(per_row), cols):
                        if idx < num:
                            movie_title = recommendations.iloc[idx]['title']
                            movie_id = recommendations.iloc[idx]['movie_id']
                            poster_url = fetch_poster(movie_id)
                            with col:
                                if poster_url:
                                    st.image(poster_url, width=150)
                                else:
                                    st.info("Poster not available")
                                st.markdown(f"**{movie_title}**")
                        else:
                            with col:
                                st.empty()
                        idx += 1



