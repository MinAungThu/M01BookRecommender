import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time


# --- Recommender Class ---
class BookRecommender:
    def __init__(self, model, book_pivot, final_rating):
        self.model = model
        self.book_pivot = book_pivot
        self.final_rating = final_rating

    def recommend_books(self, book_name):
        if book_name not in self.book_pivot.index:
            return []
        book_id = np.where(self.book_pivot.index == book_name)[0][0]
        distance , suggestions = self.model.kneighbors(self.book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        recommended_books = []
        for i in range(len(suggestions)):
            books = self.book_pivot.index[suggestions[i]]
            for j in books:
                if j != book_name:
                    recommended_books.append(j)
        return recommended_books

    def get_book_details(self, title):
        row = self.final_rating[self.final_rating['Book-Title'] == title]
        if not row.empty:
            return {
                "author": row['Book-Author'].values[0],
                "image_url": row['Image-URL-L'].values[0]
            }
        return {"author": "Unknown", "image_url": None}


# --- Load Model and Data ---
model = pickle.load(open('specs/model.pkl', 'rb'))
book_names = pickle.load(open('specs/book_names.pkl', 'rb'))
final_rating = pickle.load(open('specs/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('specs/book_pivot.pkl', 'rb'))

recommender = BookRecommender(model, book_pivot, final_rating)

# --- Streamlit UI ---
st.set_page_config(page_title="Books Recommender", layout="wide", page_icon="📚")
st.title(" M01 Book Recommendation System")
st.write("Select a book to get ML-powered suggestions!")

# --- Simple Dropdown Input ---
selected_book = st.selectbox("Choose a book you have read before:", book_names)

# --- Show Recommendations ---
if selected_book and st.button("Show Recommendations"):
    with st.spinner("Finding similar books..."):
        recommended_books = recommender.recommend_books(selected_book)
        time.sleep(2)

    if recommended_books:
        st.success(f"Books similar to: *{selected_book}*")
        cols = st.columns(5)

        for idx, book in enumerate(recommended_books):
            details = recommender.get_book_details(book)
            with cols[idx % 5]:
                st.image(details["image_url"], use_container_width=True)
                st.markdown(f"**{book}**")
                st.caption(f"By: {details['author']}")
    else:
        st.warning("No recommendations found.")
