import pandas as pd
import numpy as np
from dotenv import load_dotenv



from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


import gradio as gr

load_dotenv()

books = pd.read_csv("books_with_emotion.csv")

books = pd.read_csv("books_with_emotion.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,   # Changed from 0 to 1000
    chunk_overlap=0
)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, HuggingFaceEmbeddings())

def retrieve_semantic_recommendations(
    query: str,
    category: str = "All", # Default to All
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    # 1. Get a larger pool of candidates
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = []
    for rec in recs:
        try:
            # This logic assumes the ISBN is the first word in your text file
            isbn = int(rec.page_content.strip().split()[0])
            books_list.append(isbn)
        except (ValueError, IndexError):
            # If a line in your text file doesn't start with a number, this skips it
            continue

    # 2. Get all matching books from the dataframe
    book_recs = books[books["isbn13"].isin(books_list)].copy()

    # 3. Filter by category (Done BEFORE slicing)
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # 4. Sort by tone (Done BEFORE slicing)
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)

    # 5. Finally, return only the requested amount
    return book_recs.head(final_top_k)


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Ocean()) as dashboard:
    gr.Markdown("# Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

if __name__ == "__main__":
    dashboard.launch()



