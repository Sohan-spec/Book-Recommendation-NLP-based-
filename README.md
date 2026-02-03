# book_recommendation
This project is a semantic book recommendation system that uses natural language processing and vector search to find books based on meaning, context, and emotion rather than simple keyword matching. It was developed as an introductory machine learning project following the freeCodeCamp "Build a Semantic Book Recommender" course.

Technical Overview
The system processes a dataset of book descriptions and maps them into a high-dimensional vector space, allowing for sophisticated retrieval based on user intent.

Key Features
Natural Language Processing (NLP): Processes book descriptions to understand semantic meaning.

Text Embeddings: Uses Hugging Face sentence transformers to convert text into numerical vectors.

Vector Database: Utilizes ChromaDB to store embeddings and perform similarity searches.

Zero-Shot Classification: Automatically classifies books into categories without the need for manual labeling.

Sentiment and Emotion Analysis: Extracts emotional tones such as joy, sadness, or suspense to allow for emotion-aware filtering.

Tech Stack
Language: Python

Libraries: LangChain, Hugging Face Transformers

Data Handling: Pandas, NumPy

Storage: ChromaDB

User Interface: Gradio

Development Insights
The development process involved significant data engineering and hardware management:

Data Cleaning: Extensive use of Pandas to handle missing values, clean text data, and ensure ISBNs correctly mapped to book metadata.

Hardware Constraints: Running large models like all-mpnet-base-v2 locally required managing RAM and CPU cycles during the materialization of model weights.

Persistence: Implemented local storage for the vector database to avoid re-embedding the entire dataset on every run.

How to Use
Ensure all CSV and TXT data files are in the project directory.

Install required dependencies via pip.

Set up your api keys in .env

Run gradio-dashboard.py to launch the web interface.
