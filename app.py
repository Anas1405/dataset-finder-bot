import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
from huggingface_hub import list_datasets
import spacy
import re
import os


os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Checking user prompt for keywords and filters
def extract_keywords_and_filters(prompt):
    doc = nlp(prompt)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"] and not token.is_stop]

    filters = {
        "filetype": None,
        "size": None,
        "date": None,
        "columns": None
    }

    prompt_lower = prompt.lower()

    for ft in ["csv", "json", "xls", "xlsx", "xml"]:
        if ft in prompt_lower:
            filters["filetype"] = ft
            break

    date_match = re.search(r"(after|before)\s*(\d{4})", prompt_lower)
    if date_match:
        filters["date"] = f"{date_match.group(1)} {date_match.group(2)}"

    row_match = re.search(r"(above|more than|over|below|under|less than)\s*(\d{3,7})\s*rows?", prompt_lower)
    if row_match:
        comparison = row_match.group(1)
        number = int(row_match.group(2))
        filters["size"] = f"{comparison} {number} rows"

    column_match = re.search(r"(\d{1,3})\s*columns?", prompt_lower)
    if column_match:
        filters["columns"] = f"{column_match.group(1)} columns"

    return " ".join(keywords), filters

# search_kaggle_datasets
def search_kaggle_datasets(query, filters):
    api = KaggleApi()
    api.authenticate()
    results = api.dataset_list(search=query)
    datasets = []

    for d in results[:10]:
        if filters["filetype"] and filters["filetype"].upper() not in d.fileTypes:
            continue
        datasets.append({
            "source": "Kaggle",
            "title": d.title,
            "url": f"https://www.kaggle.com/datasets/{d.ref}",
            "size": "Unknown",  # Kaggle API does not provide size here
            "downloads": d.downloadCount if hasattr(d, "downloadCount") else "Unknown",
            "description": d.subtitle if hasattr(d, "subtitle") else "—"
        })
    return datasets


# Search Hugging Face datasets
def search_huggingface_datasets(query):
    results = list(list_datasets(search=query))
    datasets = []

    for d in results[:10]:
        desc = d.cardData.get("description", "No description") if d.cardData else "No description"
        datasets.append({
            "source": "HuggingFace",
            "title": d.id,
            "url": f"https://huggingface.co/datasets/{d.id}",
            "size": "Unknown",
            "downloads": "Unknown",
            "description": desc
        })
    return datasets

# UI settings
st.set_page_config(page_title="Dataset Finder", layout="wide")
st.title("Dataset Finder")

use_kaggle = st.checkbox("Search Kaggle", value=True)
use_hf = st.checkbox("Search Hugging Face", value=True)

user_input = st.text_input("Enter your prompt here")

if st.button("Search") and user_input:
    with st.spinner("Processing..."):
        keywords, filters = extract_keywords_and_filters(user_input)

        # Display extracted search terms
        st.markdown("### Search Keywords")
        st.markdown(f"`{keywords}`")

        # Display extracted filters in plain list
        st.markdown("### Filters")
        for key, value in filters.items():
            if value:
                st.markdown(f"- **{key.capitalize()}**: {value}")

        # Dataset results
        st.markdown("### Search Results")
        results = []

        if use_kaggle:
            results += search_kaggle_datasets(keywords, filters)
        if use_hf:
            results += search_huggingface_datasets(keywords)

        if results:
            for i, d in enumerate(results, 1):
                st.markdown(f"""
{i}. [{d['title']}]({d['url']})  
Source: {d['source']}  
Size: {d['size']}  
Downloads: {d['downloads']}  
Description: {d['description'][:300]}...
                """)
        else:
            st.warning("No datasets found. Try adjusting your query.")
