import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict, Set, Optional
import httpx
import os
import asyncio
import ssl
import certifi
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Create SSL context for requests
ssl_context = ssl.create_default_context(cafile=certifi.where())

# --- Constants - Updated to match Notion DB ---
DEFAULT_FIELDS = ["Spanish", "English", "Difficulty Level", "Category", "Reveal Answer", "Correct Answer", "Status", "Date Added"]

# Category options matching your Notion DB (with added 'Useful Phrases')
CATEGORY_OPTIONS = [
    "Culture / Media",
    "Health / Body", 
    "Politics / Economy",
    "Emotions / Relationships",
    "Travel / Tourism",
    "Academic / Education",
    "Technology",
    "Professional / Business",
    "General / Everyday",
    "Useful Phrases"  # New category added
]

# Difficulty Level options
DIFFICULTY_OPTIONS = ["Beginner", "Intermediate", "Advanced"]

# Status options
STATUS_OPTIONS = ["Not started", "Done"]

# AI API Configuration
AI_PROVIDERS = {
    "OpenRouter (Free)": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "requires_key": True
    },
    "Groq (Free)": {
        "url": "https://api.groq.com/openai/v1/chat/completions", 
        "model": "llama3-8b-8192",
        "requires_key": True
    },
    "Ollama (Local)": {
        "url": "http://localhost:11434/api/chat",
        "model": "llama3.1",
        "requires_key": False
    }
}

# --- Helper Functions ---
def get_difficulty_from_slider(slider_value: int) -> str:
    """Map slider value (1-10) to difficulty level"""
    if slider_value <= 3:
        return "Beginner"
    elif 4 <= slider_value <= 7:
        return "Intermediate"
    else:
        return "Advanced"

# --- Secrets Configuration ---
def get_secrets():
    """Get secrets from Streamlit secrets or environment variables"""
    secrets = {
        "NOTION_TOKEN": None,
        "DATABASE_ID": None,
        "GROQ_TOKEN": None,
        "OPENROUTER_TOKEN": None
    }
    
    try:
        secrets["NOTION_TOKEN"] = st.secrets.get("NOTION_TOKEN")
        secrets["DATABASE_ID"] = st.secrets.get("DATABASE_ID") 
        secrets["GROQ_TOKEN"] = st.secrets.get("GROQ_TOKEN")
        secrets["OPENROUTER_TOKEN"] = st.secrets.get("OPENROUTER_TOKEN")
    except FileNotFoundError:
        secrets["NOTION_TOKEN"] = os.environ.get("NOTION_TOKEN")
        secrets["DATABASE_ID"] = os.environ.get("DATABASE_ID")
        secrets["GROQ_TOKEN"] = os.environ.get("GROQ_TOKEN")
        secrets["OPENROUTER_TOKEN"] = os.environ.get("OPENROUTER_TOKEN")
        
    return secrets

# --- AI Vocabulary Analysis ---
async def analyze_vocabulary_with_ai(text: str, provider_config: Dict, api_key: str = None, difficulty_level: str = "Intermediate") -> List[Dict]:
    """Use AI to intelligently extract and analyze vocabulary"""
    
    prompt = f"""
You are a Spanish language learning expert. Analyze the following Spanish text and extract 15-25 of the MOST USEFUL vocabulary words for {difficulty_level} Spanish learners.

SELECTION CRITERIA:
- Focus on words that are: commonly used, educationally valuable, not too basic (avoid "el", "la", "es", "muy", etc.)
- Prioritize: nouns, adjectives, verbs, and useful phrases
- Include a mix of difficulty levels but lean toward {difficulty_level}
- Avoid proper nouns unless culturally significant
- Consider words that appear multiple times as more important

For each selected word, provide:
1. The Spanish word/phrase (exactly as it appears)
2. English translation
3. Difficulty level (Beginner/Intermediate/Advanced)
4. Category from: Culture/Media, Health/Body, Politics/Economy, Emotions/Relationships, Travel/Tourism, Academic/Education, Technology, Professional/Business, General/Everyday, Useful Phrases
5. A brief context note about why it's useful

TEXT TO ANALYZE:
{text[:3000]}

Respond in JSON format:
{{
  "vocabulary": [
    {{
      "spanish": "word",
      "english": "translation", 
      "difficulty": "{difficulty_level}",
      "category": "General / Everyday",
      "context": "Common in news articles about politics"
    }}
  ]
}}
"""

    try:
        headers = {"Content-Type": "application/json"}
        
        if provider_config["requires_key"] and api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif provider_config["requires_key"] and not api_key:
            st.error("API key required for this provider")
            return []

        # Handle different API formats
        if "ollama" in provider_config["url"]:
            # Ollama format
            payload = {
                "model": provider_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        else:
            # OpenAI-compatible format
            payload = {
                "model": provider_config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,
                "temperature": 0.3
            }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                provider_config["url"],
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract content based on API format
                if "ollama" in provider_config["url"]:
                    content = data.get("message", {}).get("content", "")
                else:
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Parse JSON response
                try:
                    # Extract JSON from response (in case there's extra text)
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        vocab_data = json.loads(json_match.group())
                        vocabulary = vocab_data.get("vocabulary", [])
                        
                        # Convert to our format matching Notion DB
                        result = []
                        for item in vocabulary:
                            result.append({
                                "Spanish": item.get("spanish", ""),
                                "English": item.get("english", ""),
                                "Difficulty Level": item.get("difficulty", difficulty_level),
                                "Category": item.get("category", "General / Everyday"),
                                "Reveal Answer": False,
                                "Correct Answer": "",  # Empty by default
                                "Status": "Not started",
                                "Date Added": datetime.today().date()  # Use date object instead of string
                            })
                        
                        return result
                    else:
                        st.error("Could not parse AI response as JSON")
                        return []
                        
                except json.JSONDecodeError as e:
                    st.error(f"JSON parsing error: {str(e)}")
                    st.text("Raw AI response:")
                    st.text(content[:500])
                    return []
            else:
                st.error(f"AI API error: {response.status_code} - {response.text}")
                return []
                
    except Exception as e:
        st.error(f"Error calling AI API: {str(e)}")
        return []

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

    # Handle Difficulty Level (select property)
    if "Difficulty Level" in row and pd.notna(row["Difficulty Level"]):
        properties["properties"]["Difficulty Level"] = {
            "select": {"name": str(row["Difficulty Level"])}
        }

    # Handle Category (select property)
    if "Category" in row and pd.notna(row["Category"]):
        properties["properties"]["Category"] = {
            "select": {"name": str(row["Category"])}
        }

    # Handle Reveal Answer (checkbox)
    if "Reveal Answer" in row and pd.notna(row["Reveal Answer"]):
        properties["properties"]["Reveal Answer"] = {
            "checkbox": bool(row["Reveal Answer"])
        }

    # Handle Correct Answer (rich text)
    if "Correct Answer" in row and pd.notna(row["Correct Answer"]):
        properties["properties"]["Correct Answer"] = {
            "rich_text": [{"text": {"content": str(row["Correct Answer"])}}]
        }

    # Handle Status (status property)
    if "Status" in row and pd.notna(row["Status"]):
        properties["properties"]["Status"] = {
            "status": {"name": str(row["Status"])}
        }

    response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=properties)
    return response.status_code == 200

# --- Async Functions ---
async def fetch_article_text(url: str) -> str:
    # First try static scraping (faster for static sites)
    static_text = await _fetch_static(url)
    if len(static_text) > 200:  # Valid static content
        return static_text

    # Fall back to dynamic scraping if static fails
    return await _fetch_dynamic(url)

async def _fetch_static(url: str) -> str:
    """Traditional scraping for static sites"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()
            return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return ""


async def _fetch_dynamic(url: str) -> str:
    """Selenium/Playwright for JS sites"""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=10000)
            content = await page.content()
            await browser.close()
            soup = BeautifulSoup(content, 'html.parser')
            text = soup.get_text()
            return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return ""

async def process_article(article_url: str, provider_config: Dict, api_key: str = None, difficulty_level: str = "Intermediate"):
    with st.spinner("Fetching article..."):
        article_text = await fetch_article_text(article_url)
        
        if not article_text:
            st.error("Could not fetch article text")
            return
            
        if len(article_text) < 200:
            st.warning("Article text seems too short. Please check the URL.")
            return
            
        st.session_state.article_text = article_text
        
        # Show preview of article
        with st.expander("Article Preview"):
            st.text_area("Extracted Article Text", 
                        value=article_text[:1000] + ("..." if len(article_text) > 1000 else ""), 
                        height=200)
    
    with st.spinner("AI is analyzing vocabulary..."):
        vocabulary = await analyze_vocabulary_with_ai(article_text, provider_config, api_key, difficulty_level)
        
        if vocabulary:
            st.session_state.vocabulary_df = pd.DataFrame(vocabulary)
            st.success(f"AI extracted {len(vocabulary)} useful vocabulary items at {difficulty_level} level!")
            
            # Show difficulty breakdown
            if vocabulary and 'Difficulty Level' in vocabulary[0]:
                difficulty_counts = pd.Series([v['Difficulty Level'] for v in vocabulary]).value_counts()
                st.write("**Difficulty Breakdown:**")
                for diff, count in difficulty_counts.items():
                    st.write(f"- {diff}: {count} words")
                    
            # Show category breakdown
            if vocabulary and 'Category' in vocabulary[0]:
                category_counts = pd.Series([v['Category'] for v in vocabulary]).value_counts()
                st.write("**Category Breakdown:**")
                for cat, count in category_counts.items():
                    st.write(f"- {cat}: {count} words")
        else:
            st.warning("AI could not extract vocabulary. Please try a different article or check your API configuration.")

# --- Streamlit App ---
def main():
    st.title("🎓 AI-Powered Spanish Vocabulary Collector")
    st.markdown("*Intelligently extract useful vocabulary from Spanish articles using AI*")

    secrets = get_secrets()

    with st.sidebar:
        st.header("⚙️ Configuration")

        # Difficulty Level Slider
        st.subheader("Difficulty Level")
        difficulty_slider = st.slider(
            "Select difficulty (1-10)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="1-3: Beginner, 4-7: Intermediate, 8-10: Advanced"
        )

        # Visual indicator
        difficulty_level = get_difficulty_from_slider(difficulty_slider)
        st.write(f"**Selected Level:** {difficulty_slider} ({difficulty_level})")

        # Notion Configuration
        st.subheader("Notion Setup")
        if secrets.get("NOTION_TOKEN"):
            notion_token = secrets["NOTION_TOKEN"]
            st.success("✅ Notion Token loaded")
        else:
            notion_token = st.text_input("Notion Token", type="password")

        if secrets.get("DATABASE_ID"):
            database_id = secrets["DATABASE_ID"]
            st.success("✅ Database ID loaded")
        else:
            database_id = st.text_input("Database ID")

        # AI Provider Configuration
        st.subheader("AI Provider")
        selected_provider = st.selectbox("Choose AI Provider", list(AI_PROVIDERS.keys()))
        provider_config = AI_PROVIDERS[selected_provider]

        # Get the appropriate API key based on provider
        ai_api_key = None
        if provider_config["requires_key"]:
            if selected_provider == "Groq (Free)":
                if secrets.get("GROQ_TOKEN"):
                    ai_api_key = secrets["GROQ_TOKEN"]
                    st.success("✅ Groq Token loaded")
                else:
                    ai_api_key = st.text_input("Groq API Key", type="password",
                                             help="Get free API key from console.groq.com")
            elif selected_provider == "OpenRouter (Free)":
                if secrets.get("OPENROUTER_TOKEN"):
                    ai_api_key = secrets["OPENROUTER_TOKEN"]
                    st.success("✅ OpenRouter Token loaded")
                else:
                    ai_api_key = st.text_input("OpenRouter API Key", type="password",
                                             help="Get free credits from openrouter.ai")
            else:
                ai_api_key = st.text_input(f"{selected_provider} API Key", type="password")
        else:
            st.info("Local Ollama - no API key needed")

        # Test Connections
        if st.button("🔧 Test Connections"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Notion:**")
                if notion_token and database_id:
                    try:
                        existing_words = get_existing_words(notion_token, database_id)
                        st.success(f"✅ Connected ({len(existing_words)} words)")
                    except Exception as e:
                        st.error(f"❌ Failed: {str(e)}")
                else:
                    st.warning("⚠️ Credentials missing")

            with col2:
                st.write(f"**{selected_provider}:**")
                if not provider_config["requires_key"] or ai_api_key:
                    st.success("✅ Ready")
                else:
                    st.warning("⚠️ API key needed")

    # Main tabs
    tab1, tab2 = st.tabs(["📰 From Article URL", "✏️ Manual Entry"])

    with tab1:
        st.header("Extract Vocabulary from Spanish Article")

        article_url = st.text_input("🔗 Enter Spanish Article URL", placeholder="https://elpais.com/...")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🤖 Extract with AI", type="primary"):
                if not article_url:
                    st.warning("Please enter a URL")
                elif provider_config["requires_key"] and not ai_api_key:
                    st.warning(f"Please configure {selected_provider} API key")
                else:
                    difficulty_level = get_difficulty_from_slider(difficulty_slider)
                    asyncio.run(process_article(article_url, provider_config, ai_api_key, difficulty_level))
    with tab2:
        st.header("Manually Add Vocabulary")

        # Initialize DataFrame with proper columns and types if it doesn't exist
        if 'vocabulary_df' not in st.session_state:
            st.session_state.vocabulary_df = pd.DataFrame({
                "Spanish": pd.Series(dtype='str'),
                "English": pd.Series(dtype='str'),
                "Difficulty Level": pd.Series(dtype='str'),
                "Category": pd.Series(dtype='str'),
                "Reveal Answer": pd.Series(dtype='bool'),
                "Correct Answer": pd.Series(dtype='str'),
                "Status": pd.Series(dtype='str'),
                "Date Added": pd.Series(dtype='datetime64[ns]')
            })

        # Ensure all columns exist and have correct types
        df = st.session_state.vocabulary_df
        for col in DEFAULT_FIELDS:
            if col not in df.columns:
                if col == "Reveal Answer":
                    df[col] = False
                elif col == "Date Added":
                    df[col] = pd.to_datetime(datetime.today().date())
                else:
                    df[col] = ""

        # Convert date column to datetime if it's not already
        if 'Date Added' in df:
            df['Date Added'] = pd.to_datetime(df['Date Added'])

        # Create the data editor with robust column configuration
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            column_config={
                "Spanish": st.column_config.TextColumn(
                    "Spanish",
                    required=True,
                    default=""
                ),
                "English": st.column_config.TextColumn(
                    "English",
                    required=True,
                    default=""
                ),
                "Difficulty Level": st.column_config.SelectboxColumn(
                    "Difficulty Level",
                    options=DIFFICULTY_OPTIONS,
                    default="Intermediate"
                ),
                "Category": st.column_config.SelectboxColumn(
                    "Category",
                    options=CATEGORY_OPTIONS,
                    default="General / Everyday"
                ),
                "Correct Answer": st.column_config.TextColumn(
                    "Correct Answer",
                    default=""
                ),
                "Reveal Answer": st.column_config.CheckboxColumn(
                    "Reveal Answer",
                    default=False
                ),
                "Status": st.column_config.SelectboxColumn(
                    "Status",
                    options=STATUS_OPTIONS,
                    default="Not started"
                ),
                "Date Added": st.column_config.DateColumn(
                    "Date Added",
                    format="YYYY-MM-DD",
                    default=datetime.today().date()
                )
            },
            hide_index=True,
            use_container_width=True
        )

        if st.button("💾 Update Vocabulary List"):
            st.session_state.vocabulary_df = edited_df
            st.success("Vocabulary list updated!")

    # Review and Push Section
    if 'vocabulary_df' in st.session_state and not st.session_state.vocabulary_df.empty:
        st.divider()
        st.header("📋 Review & Push to Notion")

        # Show summary
        df = st.session_state.vocabulary_df
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Words", len(df))
        with col2:
            if 'Difficulty Level' in df.columns:
                intermediate_count = len(df[df['Difficulty Level'] == 'Intermediate'])
                st.metric("Intermediate", intermediate_count)
        with col3:
            if 'Category' in df.columns:
                most_common_cat = df['Category'].mode().iloc[0] if not df['Category'].empty else "N/A"
                st.metric("Most Common Category", most_common_cat)

        # Show the data
        st.dataframe(df, use_container_width=True)

        if st.button("🚀 Push to Notion", type="primary"):
            if not notion_token or not database_id:
                st.warning("Please configure Notion connection")
            else:
                with st.spinner("Pushing to Notion..."):
                    try:
                        # Get existing words in lowercase for comparison
                        existing_words = {word.lower() for word in get_existing_words(notion_token, database_id)}

                        # Clean and filter the DataFrame
                        df_clean = df.dropna(subset=["Spanish"])
                        df_clean = df_clean[df_clean["Spanish"].str.strip() != ""]  # Remove empty strings

                        # Check for duplicates (case-insensitive)
                        df_clean["is_duplicate"] = df_clean["Spanish"].str.lower().isin(existing_words)
                        duplicates = df_clean[df_clean["is_duplicate"]]
                        new_words = df_clean[~df_clean["is_duplicate"]]

                        if not duplicates.empty:
                            st.warning(f"Found {len(duplicates)} duplicates that won't be added:")
                            st.dataframe(duplicates[["Spanish", "English"]])

                        if new_words.empty:
                            st.warning("No new words to add after duplicate check")
                        else:
                            progress_bar = st.progress(0)
                            success_count = 0

                            for i, row in enumerate(new_words.to_dict("records")):
                                if create_page(row, notion_token, database_id):
                                    success_count += 1
                                progress_bar.progress((i + 1) / len(new_words))
                                time.sleep(0.3)  # Rate limiting

                            st.success(f"🎉 Successfully added {success_count} new words to Notion!")
                            st.balloons()

                            # Clear the vocabulary after successful push
                            if st.button("Clear Vocabulary List"):
                                empty_data = {col: [] for col in DEFAULT_FIELDS}
                                st.session_state.vocabulary_df = pd.DataFrame(empty_data)
                                st.rerun()

                    except Exception as e:
                        st.error(f"Error pushing to Notion: {str(e)}")

if __name__ == "__main__":
    main()
