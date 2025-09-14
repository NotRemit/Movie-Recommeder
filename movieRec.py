import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import kagglehub
from rapidfuzz import process
import random

# -------------------
# Load dataset
# -------------------
path = kagglehub.dataset_download("abhikjha/movielens-100k")
ratings_path = f"{path}/ml-latest-small/ratings.csv"
movies_path = f"{path}/ml-latest-small/movies.csv"

df = pd.read_csv(ratings_path)
movies_name = pd.read_csv(movies_path)

df = pd.pivot(df, index='movieId', columns='userId', values='rating')
movies_ids = df.fillna(0)
similarity_score = cosine_similarity(movies_ids)

# -------------------
# Caching for posters
# -------------------
poster_cache = {}

# -------------------
# Functions
# -------------------
def recommendedMoviesList(title):
    mId = movies_name.loc[movies_name['title'] == title, 'movieId'].iloc[0] #type: ignore
    index = np.where(movies_ids.index == mId)[0][0]
    suggested_movies_index = []
    suggested_movies = []
    similar_movies = sorted(
        list(enumerate(similarity_score[index])),
        key=lambda x: x[1], reverse=True
    )[1:6]
    for movie in similar_movies:
        suggested_movies_index.append(movie[0])
    for idx in suggested_movies_index:
        movie_id = movies_ids.index[idx]
        movie_name = movies_name.loc[movies_name['movieId'] == movie_id, 'title'].iloc[0]
        suggested_movies.append(movie_name)
    return suggested_movies

API_KEY = st.secrets["OMDB_API_KEY"]

def get_movie_data(title, year=None):
    base_url = "http://www.omdbapi.com/"
    params = {
        "apikey": API_KEY,
        "t": title,
    }
    if year:
        params["y"] = year
    response = requests.get(base_url, params=params)
    return response.json()

def get_poster_url(title, year=None):
    if title in poster_cache:
        return poster_cache[title]
    
    data = get_movie_data(title, year)
    if data.get("Response") == "True" and data.get("Poster") != "N/A":
        poster_url = data["Poster"]
        poster_cache[title] = poster_url
        return poster_url
    
    fallback_url = "https://ih1.redbubble.net/image.1027712254.9762/fposter,small,wall_texture,product,750x1000.u2.jpg"  # fallback image
    poster_cache[title] = fallback_url
    return fallback_url

def split_title_year(movie):
    if '(' in movie and ')' in movie:
        title_part = movie.rsplit('(', 1)[0].strip()
        year_part = movie.rsplit('(', 1)[1].replace(')', '').strip()
    else:
        title_part = movie
        year_part = None
    return title_part, year_part

def fuzzy_match_movie(query, choices, limit=5, score_cutoff=60):
    matches = process.extract(query, choices, limit=limit, score_cutoff=score_cutoff)
    return matches

# --- Calculate Top 10 Rated Movies ---
average_ratings = movies_ids.mean(axis=1) 
counts = (movies_ids != 0).sum(axis=1)

ratings_summary = pd.DataFrame({
    'movieId': average_ratings.index,
    'average_rating': average_ratings.values,
    'num_ratings': counts.values
})

movies_with_ratings = ratings_summary.merge(movies_name, on='movieId')
movies_with_ratings_filtered = movies_with_ratings[movies_with_ratings['num_ratings'] >= 50]
top10_movies = movies_with_ratings_filtered.sort_values('average_rating', ascending=False).head(10)

top10_titles = top10_movies['title'].tolist()

top10_posters = [
    'https://m.media-amazon.com/images/M/MV5BMDAyY2FhYjctNDc5OS00MDNlLThiMGUtY2UxYWVkNGY2ZjljXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg',
    'https://m.media-amazon.com/images/M/MV5BNDYwNzVjMTItZmU5YS00YjQ5LTljYjgtMjY2NDVmYWMyNWFmXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg',
    'https://m.media-amazon.com/images/M/MV5BYTViYTE3ZGQtNDBlMC00ZTAyLTkyODMtZGRiZDg0MjA2YThkXkEyXkFqcGc@._V1_.jpg',
    'https://m.media-amazon.com/images/M/MV5BN2NmN2VhMTQtMDNiOS00NDlhLTliMjgtODE2ZTY0ODQyNDRhXkEyXkFqcGc@._V1_.jpg',
    'https://m.media-amazon.com/images/M/MV5BNDdhOGJhYzctYzYwZC00YmI2LWI0MjctYjg4ODdlMDExYjBlXkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg',
    'https://m.media-amazon.com/images/M/MV5BZTQ2MDhmMWMtZjk4Ni00ZDM1LWFjNGEtYzhkNWRmMjk1NzI0XkEyXkFqcGc@._V1_.jpg',
    'https://m.media-amazon.com/images/M/MV5BNGMxZDBhNGQtYTZlNi00N2UzLWI4NDEtNmUzNWM2NTdmZDA0XkEyXkFqcGc@._V1_.jpg',
    'https://m.media-amazon.com/images/M/MV5BOTgyOGQ1NDItNGU3Ny00MjU3LTg2YWEtNmEyYjBiMjI1Y2M5XkEyXkFqcGc@._V1_FMjpg_UX1000_.jpg',
    'https://m.media-amazon.com/images/M/MV5BNjM1ZDQxYWUtMzQyZS00MTE1LWJmZGYtNGUyNTdlYjM3ZmVmXkEyXkFqcGc@._V1_.jpg',
    'https://i.pinimg.com/736x/51/ba/64/51ba64b2e61f820e0e86bdd2f4c6e92c.jpg'
]

# -------------------
# Streamlit UI
# -------------------
st.title("🎬 Movie Recommender System")


search_movie = st.text_input("Search for a movie", "")

if search_movie:

    titles_list = movies_name['title'].tolist()
    matches = fuzzy_match_movie(search_movie, titles_list)

    if matches:
        options = [m[0] for m in matches]
        matched_title = st.selectbox("Choose the closest match:", options)

    
        t, y = split_title_year(matched_title)
        main_poster = get_poster_url(t, y)
        st.image(main_poster, width=200)

        st.write("### Recommended Movies:")
        showMovies = recommendedMoviesList(matched_title)
        cols = st.columns(5)
        for i, movie in enumerate(showMovies):
            with cols[i]:
                t, y = split_title_year(movie)
                poster_url = get_poster_url(t, y)
                st.image(poster_url, width=150)
                st.caption(movie)
    else:
        st.warning("No similar movie found.")
else:
    st.write("### 🏆 Top 10 Rated Movies")
    

    cols1 = st.columns(5)
    for i in range(5):
        with cols1[i]:
            st.image(top10_posters[i], width=150)
            st.caption(top10_titles[i])
    

    cols2 = st.columns(5)
    for i in range(5):

        with cols2[i]:
            st.image(top10_posters[i+5], width=150)
            st.caption(top10_titles[i+5])