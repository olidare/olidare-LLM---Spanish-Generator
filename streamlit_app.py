import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict, Set
import httpx
import os
import asyncio

# --- Constants ---
DEFAULT_FIELDS = ["Spanish", "English", "Date Added", "Reveal Answer", "Status", "Correct Answer"]
HF_MODEL = "Helsinki-NLP/opus-mt-es-en"

# --- Secrets Configuration ---
def get_secrets():
    """Get secrets from Streamlit secrets or environment variables"""
    secrets = {
        "NOTION_TOKEN": None,
        "DATABASE_ID": None,
        "HF_TOKEN": None
    }
    
    try:
        secrets["NOTION_TOKEN"] = st.secrets.get("NOTION_TOKEN")
        secrets["DATABASE_ID"] = st.secrets.get("DATABASE_ID")
        secrets["HF_TOKEN"] = st.secrets.get("HF_TOKEN")
    except FileNotFoundError:
        secrets["NOTION_TOKEN"] = os.environ.get("NOTION_TOKEN")
        secrets["DATABASE_ID"] = os.environ.get("DATABASE_ID")
        secrets["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        
    return secrets

# --- Notion API Functions ---
@st.cache_data(ttl=3600)
def get_existing_words(notion_token: str, database_id: str, check_field: str = "Spanish") -> Set[str]:
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    
    existing_words = set()
    has_more = True
    next_cursor = None

    while has_more:
        url = f"https://api.notion.com/v1/databases/{database_id}/query"
        payload = {"page_size": 100}
        if next_cursor:
            payload["start_cursor"] = next_cursor

        response = requests.post(url, headers=headers, json=payload)
        data = response.json()

        for result in data.get("results", []):
            spanish_prop = result["properties"].get(check_field)
            if spanish_prop and spanish_prop["type"] == "title":
                text = spanish_prop["title"]
                if text:
                    content = text[0]["text"]["content"]
                    existing_words.add(content.strip().lower())

        has_more = data.get("has_more", False)
        next_cursor = data.get("next_cursor")

    return existing_words

def create_page(row: Dict, notion_token: str, database_id: str) -> bool:
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28",
    }
    
    today = datetime.today().strftime("%Y-%m-%d")

    properties = {
        "parent": { "database_id": database_id },
        "properties": {
            "Spanish": {
                "title": [{ "text": { "content": row["Spanish"] } }]
            },
            "English": {
                "rich_text": [{ "text": { "content": str(row["English"]) } }]
            },
            "Date Added": {
                "date": { "start": today }
            }
        }
    }

    if "Reveal Answer" in row:
        properties["properties"]["Reveal Answer"] = {
            "checkbox": bool(row["Reveal Answer"]) if pd.notna(row["Reveal Answer"]) else False
        }

    if "Status" in row:
        properties["properties"]["Status"] = {
            "status": {
                "name": str(row["Status"]) if pd.notna(row["Status"]) else "Not started"
            }
        }

    if "Correct Answer" in row and pd.notna(row["Correct Answer"]):
        properties["properties"]["Correct Answer"] = {
            "rich_text": [{ "text": { "content": str(row["Correct Answer"]) } }]
        }

    response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=properties)
    return response.status_code == 200

# --- Async Functions ---
async def fetch_article_text(url: str) -> str:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            soup = BeautifulSoup(response.text, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'img']):
                element.decompose()
            text = soup.get_text()
            return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        st.error(f"Error fetching article: {str(e)}")
        return ""

async def extract_vocabulary_with_hf(text: str, hf_token: str) -> List[Dict]:
    try:
        API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        headers = {"Authorization": f"Bearer {hf_token}"}
        spanish_words = list(set(re.findall(r'\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]{5,}\b', text[:2000])))
        
        translations = []
        async with httpx.AsyncClient() as client:
            for word in spanish_words[:20]:
                response = await client.post(
                    API_URL,
                    headers=headers,
                    json={"inputs": word},
                    timeout=30.0
                )
                if response.status_code == 200:
                    translations.append({
                        "Spanish": word,
                        "English": response.json()[0]['translation_text'],
                        "Reveal Answer": False,
                        "Status": "Not started"
                    })
                else:
                    st.warning(f"Couldn't translate '{word}': {response.text}")
        return translations
    except Exception as e:
        st.error(f"HF API Error: {str(e)}")
        return []

async def process_article(article_url: str, hf_token: str):
    with st.spinner("Processing article..."):
        article_text = await fetch_article_text(article_url)
        if article_text:
            st.session_state.article_text = article_text
            st.text_area("Extracted Article Text", 
                        value=article_text[:1000] + ("..." if len(article_text) > 1000 else ""), 
                        height=200)
            
            vocabulary = await extract_vocabulary_with_hf(article_text, hf_token)
            if vocabulary:
                st.session_state.vocabulary_df = pd.DataFrame(vocabulary)
                st.success(f"Found {len(vocabulary)} vocabulary items")
            else:
                st.warning("No vocabulary could be extracted")
        else:
            st.error("Could not fetch article text")

# --- Streamlit App ---
def main():
    st.title("Spanish Vocabulary Collector")
    secrets = get_secrets()
    
    with st.sidebar:
        st.header("Configuration")
        
        if secrets.get("NOTION_TOKEN"):
            notion_token = secrets["NOTION_TOKEN"]
            if st.toggle("Show Notion Token", False):
                st.text_input("Notion Token", value=f"{notion_token[:4]}...{notion_token[-4:]}", disabled=True)
            else:
                st.success("✅ Notion Token loaded")
        else:
            notion_token = st.text_input("Notion Token", type="password")
        
        if secrets.get("DATABASE_ID"):
            database_id = secrets["DATABASE_ID"]
            st.text_input("Database ID", value="************", disabled=True)
        else:
            database_id = st.text_input("Database ID")
        
        if secrets.get("HF_TOKEN"):
            hf_token = secrets["HF_TOKEN"]
            if st.toggle("Show HF Token", False):
                st.text_input("HF Token", value=f"{hf_token[:4]}...{hf_token[-4:]}", disabled=True)
            else:
                st.success("✅ HF Token loaded")
        else:
            hf_token = st.text_input("HuggingFace Token", type="password")
        
        if st.button("Test Connections"):
            col1, col2 = st.columns(2)
            with col1:
                if notion_token and database_id:
                    try:
                        existing_words = get_existing_words(notion_token, database_id)
                        st.success(f"✅ Notion: {len(existing_words)} words")
                    except Exception as e:
                        st.error(f"❌ Notion: {str(e)}")
            
            with col2:
                if hf_token:
                    try:
                        test = requests.get(
                            "https://huggingface.co/api/whoami",
                            headers={"Authorization": f"Bearer {hf_token}"},
                            timeout=5
                        )
                        if test.status_code == 200:
                            st.success(f"✅ HF: {test.json()['name']}")
                        else:
                            st.error(f"❌ HF: Invalid token")
                    except Exception as e:
                        st.error(f"❌ HF: Connection failed")

    tab1, tab2 = st.tabs(["From Article URL", "Manual Entry"])
    
    with tab1:
        st.header("Extract Vocabulary from Article")
        article_url = st.text_input("Enter Spanish Article URL")
        
        if st.button("Extract Vocabulary"):
            if not article_url:
                st.warning("Please enter a URL")
            elif not hf_token:
                st.warning("Please configure HuggingFace token")
            else:
                asyncio.run(process_article(article_url, hf_token))
    
    with tab2:
        st.header("Manually Add Vocabulary")
        if 'vocabulary_df' not in st.session_state:
            st.session_state.vocabulary_df = pd.DataFrame(columns=DEFAULT_FIELDS)
        
        edited_df = st.data_editor(
            st.session_state.vocabulary_df,
            num_rows="dynamic",
            column_config={
                "Spanish": st.column_config.TextColumn(required=True),
                "English": st.column_config.TextColumn(required=True),
                "Reveal Answer": st.column_config.CheckboxColumn(default=False),
                "Status": st.column_config.SelectboxColumn(
                    options=["Not started", "Learning", "Mastered"],
                    default="Not started"
                ),
                "Correct Answer": st.column_config.TextColumn()
            }
        )
        
        if st.button("Update Vocabulary List"):
            st.session_state.vocabulary_df = edited_df
    
    if 'vocabulary_df' in st.session_state and not st.session_state.vocabulary_df.empty:
        st.divider()
        st.header("Review & Push to Notion")
        st.dataframe(st.session_state.vocabulary_df)
        
        if st.button("Push to Notion"):
            if not notion_token or not database_id:
                st.warning("Please configure Notion connection")
            else:
                with st.spinner("Processing..."):
                    try:
                        existing_words = get_existing_words(notion_token, database_id)
                        df = st.session_state.vocabulary_df
                        df = df.dropna(subset=["Spanish"])
                        new_rows = df[~df["Spanish"].str.lower().isin(existing_words)]
                        
                        if new_rows.empty:
                            st.warning("All words already exist in Notion")
                            return
                        
                        progress_bar = st.progress(0)
                        success_count = 0
                        
                        for i, row in enumerate(new_rows.to_dict("records")):
                            if create_page(row, notion_token, database_id):
                                success_count += 1
                            progress_bar.progress((i + 1) / len(new_rows))
                            time.sleep(0.3)
                        
                        st.success(f"Added {success_count} new words to Notion!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
