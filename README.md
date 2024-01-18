# Book Recommendation System

## Overview

This project implements a Book Recommendation System using Flask, collaborative filtering, and popularity-based approaches. The system allows users to view popular books and receive personalized book recommendations based on collaborative filtering.

## Files

- **Main.py:** Python script containing the data processing, model building, and popularity-based recommendation logic.
- **App.py:** Flask web application script for serving the recommendation system.
- **templates:** Folder containing HTML templates for the web application.
- **popular.pkl:** Pickle file storing the popular books dataframe.
- **pt.pkl:** Pickle file storing the collaborative filtering pivot table.
- **books.pkl:** Pickle file storing book data.
- **similarity_scores.pkl:** Pickle file storing the cosine similarity scores for collaborative filtering.

## Setup

1. Ensure you have Python installed on your system.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the `Main.py` script to preprocess data, build models, and save the necessary files.
4. Execute the `App.py` script to start the Flask web application.

## Usage

- Access the web application by navigating to `http://localhost:5001/` in your web browser.
- The homepage displays a list of popular books.
- Navigate to `http://localhost:5001/recommend` to access the recommendation form.
- Enter a book title in the form to receive personalized recommendations.

## Notes

- The recommendation system is based on both popularity and collaborative filtering.
- Popularity-based recommendations are generated for popular books with a minimum number of ratings.
- Collaborative filtering provides personalized recommendations based on user behavior.

## Author

Hussain Burhanuddin
