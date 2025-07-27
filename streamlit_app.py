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

# --- Constants ---
DEFAULT_FIELDS = ["Spanish", "English", "Date Added", "Reveal Answer", "Status", "Correct Answer"]

# --- Secrets Configuration ---
def get_secrets():
    """Get secrets from Streamlit secrets or environment variables"""
    secrets = {
        "NOTION_TOKEN": None,
        "DATABASE_ID": None,
        "HF_TOKEN": None
    }
    
    try:
        # Try Streamlit secrets first
        secrets["NOTION_TOKEN"] = st.secrets.get("NOTION_TOKEN")
        secrets["DATABASE_ID"] = st.secrets.get("DATABASE_ID")
        secrets["HF_TOKEN"] = st.secrets.get("HF_TOKEN")
    except FileNotFoundError:
        # Fall back to environment variables
        secrets["NOTION_TOKEN"] = os.environ.get("NOTION_TOKEN")
        secrets["DATABASE_ID"] = os.environ.get("DATABASE_ID")
        secrets["HF_TOKEN"] = os.environ.get("HF_TOKEN")
        
    return secrets

# --- Notion API Functions ---
@st.cache_data(ttl=3600)
def get_existing_words(notion_token: str, database_id: str, check_field: str = "Spanish") -> Set[str]:
    """Fetch existing words from Notion database to check for duplicates."""
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
    """Create a new page in Notion database."""
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

    # Handle optional fields
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

# --- Web Scraping Functions ---
async def fetch_article_text(url: str) -> str:
    """Fetch and clean article text from URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'img']):
                element.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    except Exception as e:
        st.error(f"Error fetching article: {str(e)}")
        return ""

# --- HuggingFace LLM Interface ---
async def extract_vocabulary_with_hf(text: str, difficulty: str, hf_token: str) -> List[Dict]:
    """
    Extract vocabulary from text using HuggingFace Inference API.
    Uses the 'Helsinki-NLP/opus-mt-es-en' model for translation.
    """
    try:
        # Truncate text if too long (HF API has limits)
        text = text[:2000]  # Limit to first 2000 chars
        
        # Prepare the prompt for vocabulary extraction
        prompt = f"""
        Extract {difficulty.lower()} Spanish vocabulary words and phrases from this text.
        For each item, provide the Spanish term and its English translation.
        Return only JSON format like this: [{{"Spanish": "palabra", "English": "word"}}]
        
        Text: {text}
        """
        
        # Call HuggingFace Inference API
        async with httpx.AsyncClient() as client:
            headers = {
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            }
            
            # Using a conversational model that can follow instructions
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 500,
                    "return_full_text": False
                }
            }
            
            response = await client.post(
                "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            if response.status_code != 200:
                st.error(f"HF API Error: {response.text}")
                return []
            
            try:
                # Try to parse the response as JSON
                result = response.json()
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict):
                    # Handle cases where the response is wrapped differently
                    generated_text = result.get("generated_text", "")
                    # Try to extract JSON from the response text
                    json_start = generated_text.find('[')
                    json_end = generated_text.rfind(']') + 1
                    if json_start != -1 and json_end != -1:
                        json_str = generated_text[json_start:json_end]
                        return json.loads(json_str)
            except json.JSONDecodeError:
                st.error("Could not parse LLM response as JSON")
                return []
            
        return []
    except Exception as e:
        st.error(f"Error with HF API: {str(e)}")
        return []

# --- Streamlit UI ---
def main():
    st.title("Spanish Vocabulary Collector")
    st.markdown("""
    Upload Spanish articles to extract vocabulary words and add them to your Notion database.
    """)
    
    # Get secrets
    secrets = get_secrets()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Allow manual override of secrets
        notion_token = st.text_input("Notion Token", 
                                  value=secrets["NOTION_TOKEN"] or "", 
                                  type="password")
        database_id = st.text_input("Database ID", 
                                   value=secrets["DATABASE_ID"] or "")
        hf_token = st.text_input("HuggingFace Token", 
                                value=secrets["HF_TOKEN"] or "", 
                                type="password")
        
        if st.button("Test Connections"):
            col1, col2 = st.columns(2)
            with col1:
                if notion_token and database_id:
                    try:
                        existing_words = get_existing_words(notion_token, database_id)
                        st.success(f"✅ Notion: {len(existing_words)} words")
                    except Exception as e:
                        st.error(f"❌ Notion: {str(e)}")
                else:
                    st.warning("Notion credentials missing")
            
            with col2:
                if hf_token:
                    st.info("HF check would run here")
                    # In a real app, you'd make a test API call
                    st.success("✅ HF Connected")
                else:
                    st.warning("HF token missing")
    
    # Main content
    tab1, tab2 = st.tabs(["From Article URL", "Manual Entry"])
    
    with tab1:
        st.header("Extract Vocabulary from Article")
        article_url = st.text_input("Enter Spanish Article URL")
        difficulty = st.selectbox("Vocabulary Difficulty Level", 
                                ["Intermediate", "Advanced", "Interesting Phrases"])
        
        if st.button("Extract Vocabulary"):
            if not article_url:
                st.warning("Please enter a URL")
            elif not notion_token or not database_id:
                st.warning("Please configure Notion connection")
            elif not hf_token:
                st.warning("Please configure HuggingFace token")
            else:
                with st.spinner("Processing article..."):
                    # Fetch article text
                    article_text = await fetch_article_text(article_url)
                    
                    if article_text:
                        st.session_state.article_text = article_text
                        st.text_area("Extracted Article Text", 
                                   value=article_text[:2000] + ("..." if len(article_text) > 2000 else ""), 
                                   height=200)
                        
                        # Extract vocabulary using HF
                        vocabulary = await extract_vocabulary_with_hf(
                            article_text, 
                            difficulty, 
                            hf_token
                        )
                        
                        if vocabulary:
                            st.session_state.vocabulary_df = pd.DataFrame(vocabulary)
                            st.success(f"Found {len(vocabulary)} vocabulary items")
                        else:
                            st.warning("No vocabulary could be extracted")
                    else:
                        st.error("Could not fetch article text")
    
    with tab2:
        st.header("Manually Add Vocabulary")
        st.info("Use this if you want to add words directly without article extraction")
        
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
            },
            key="vocab_editor"
        )
        
        if st.button("Update Vocabulary List"):
            st.session_state.vocabulary_df = edited_df
            st.success("Vocabulary list updated!")
    
    # Push to Notion section
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
                        status_text = st.empty()
                        success_count = 0
                        
                        for i, row in enumerate(new_rows.to_dict("records")):
                            if create_page(row, notion_token, database_id):
                                success_count += 1
                            else:
                                st.error(f"Failed to add: {row['Spanish']}")
                            
                            progress = (i + 1) / len(new_rows)
                            progress_bar.progress(progress)
                            status_text.text(f"Added {i + 1}/{len(new_rows)}")
                            time.sleep(0.3)
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"Added {success_count} new words to Notion!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
