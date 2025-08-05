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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import docx
import PyPDF2
from io import BytesIO

# Create SSL context for requests
ssl_context = ssl.create_default_context(cafile=certifi.where())

# --- Constants - Updated to include phrases ---
DEFAULT_FIELDS = ["Spanish", "English", "Type", "Difficulty Level", "Category", "Reveal Answer", "Correct Answer", "Status", "Date Added"]

# Type options (new field to distinguish words from phrases)
TYPE_OPTIONS = ["Word", "Phrase"]

# Category options matching your Notion DB (with added categories for phrases)
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
    "Useful Phrases",
    "Idioms / Expressions",
    "Conversational Phrases",
    "Grammar Structures"
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

# Supported file types
SUPPORTED_FILE_TYPES = ['txt', 'pdf', 'docx', 'doc']


# --- Helper Functions ---
def get_difficulty_from_slider(slider_value: int) -> str:
    """Map slider value (1-10) to difficulty level"""
    if slider_value <= 3:
        return "Beginner"
    elif 4 <= slider_value <= 7:
        return "Intermediate"
    else:
        return "Advanced"


def is_likely_proper_noun(word: str) -> bool:
    """Check if a word is likely a proper noun that should be filtered out"""
    word_lower = word.lower().strip()
    
    # Common countries (extend this list as needed)
    countries = {
        'españa', 'francia', 'alemania', 'italia', 'portugal', 'brasil', 'argentina', 
        'chile', 'colombia', 'méxico', 'venezuela', 'perú', 'ecuador', 'bolivia',
        'uruguay', 'paraguay', 'costa rica', 'guatemala', 'honduras', 'nicaragua',
        'panamá', 'república dominicana', 'cuba', 'puerto rico', 'estados unidos',
        'reino unido', 'china', 'japón', 'india', 'rusia', 'canadá', 'australia'
    }
    
    # Common cities
    cities = {
        'madrid', 'barcelona', 'valencia', 'sevilla', 'zaragoza', 'málaga', 'murcia',
        'palma', 'bilbao', 'alicante', 'córdoba', 'valladolid', 'vigo', 'gijón',
        'hospitalet', 'vitoria', 'granada', 'elche', 'oviedo', 'badalona', 'cartagena',
        'terrassa', 'jerez', 'sabadell', 'móstoles', 'santa cruz', 'pamplona', 'almería',
        'parís', 'londres', 'berlín', 'roma', 'lisboa', 'buenos aires', 'bogotá',
        'lima', 'santiago', 'caracas', 'quito', 'montevideo', 'asunción'
    }
    
    # Tech tools and programming languages
    tech_terms = {
        'python', 'javascript', 'java', 'html', 'css', 'sql', 'php', 'ruby',
        'swift', 'kotlin', 'react', 'angular', 'vue', 'node', 'django', 'flask',
        'streamlit', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'git', 'github',
        'docker', 'kubernetes', 'aws', 'azure', 'google cloud', 'linux', 'windows',
        'macos', 'android', 'ios', 'mysql', 'postgresql', 'mongodb', 'redis'
    }
    
    # Common brands and companies
    brands = {
        'google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta', 'twitter',
        'instagram', 'whatsapp', 'telegram', 'spotify', 'netflix', 'youtube',
        'linkedin', 'tiktok', 'uber', 'airbnb', 'paypal', 'visa', 'mastercard',
        'coca cola', 'pepsi', 'mcdonalds', 'nike', 'adidas', 'samsung', 'sony'
    }
    
    # Check if word is in any of our exclusion lists
    if word_lower in countries or word_lower in cities or word_lower in tech_terms or word_lower in brands:
        return True
    
    # Check if word starts with capital letter (likely proper noun)
    if word and word[0].isupper() and len(word) > 2:
        # But allow some common words that might be capitalized at start of sentence
        common_capitalized = {'como', 'cuando', 'donde', 'porque', 'aunque', 'mientras', 'durante'}
        if word_lower not in common_capitalized:
            return True
    
    return False


def get_enhanced_prompt(text: str, difficulty_level: str) -> str:
    """Enhanced prompt to extract both words and phrases"""
    return f"""
You are a Spanish language learning expert. Analyze the following Spanish text and extract 20-30 of the MOST USEFUL vocabulary items (both individual words AND phrases) for {difficulty_level} Spanish learners.

EXTRACT BOTH:
1. **INDIVIDUAL WORDS**: Important verbs, nouns, adjectives, adverbs
2. **USEFUL PHRASES**: Common expressions, idioms, collocations, grammar structures (2-6 words)

CRITICAL EXCLUSION RULES - DO NOT INCLUDE:
- Proper nouns: Names of people, countries, cities, regions, organizations
- Brand names: Google, Microsoft, Apple, Netflix, etc.
- Technical terms: Python, JavaScript, HTML, software names, etc.
- Acronyms and abbreviations: EU, USA, GDP, etc.
- Numbers and dates as words
- Very basic words: el, la, es, muy, de, en, con, por, para, que, se, un, una

PHRASE SELECTION CRITERIA:
- Common collocations (e.g., tener en cuenta, de vez en cuando)
- Useful expressions (e.g., por lo tanto, sin embargo)
- Grammar structures (e.g., no solo... sino también)
- Conversational phrases (e.g., qué tal, de nada)
- Idiomatic expressions (e.g., estar en las nubes)

WORD SELECTION CRITERIA:
- Common verbs, nouns, adjectives that Spanish learners need
- Words used in daily conversation and practical situations
- Academic or professional vocabulary appropriate for the level
- Words that appear multiple times in the text (indicating importance)
- Vocabulary that helps express ideas, emotions, or describe situations

DIFFICULTY GUIDELINES:
- Beginner: Essential everyday words and basic phrases
- Intermediate: More complex vocabulary and common expressions
- Advanced: Sophisticated vocabulary, complex phrases, and nuanced expressions

CATEGORIZATION:
- Words: Use traditional categories (Culture/Media, Health/Body, etc.)
- Phrases: Use Useful Phrases, Idioms / Expressions, Conversational Phrases, or Grammar Structures

For each selected item, provide:
1. The Spanish word/phrase (exactly as it appears, in lowercase unless it's a legitimate proper adjective)
2. Clear, concise English translation
3. Type: Word or Phrase
4. Difficulty level (Beginner/Intermediate/Advanced)
5. Most appropriate category
6. Brief context about why it's educationally valuable

TEXT TO ANALYZE:
{text[:4000]}

Respond in JSON format:
{{
  "vocabulary": [
    {{
      "spanish": "tener en cuenta",
      "english": "to take into account",
      "type": "Phrase",
      "difficulty": "{difficulty_level}",
      "category": "Useful Phrases",
      "context": "Common expression used in formal and informal contexts"
    }},
    {{
      "spanish": "importante",
      "english": "important",
      "type": "Word", 
      "difficulty": "{difficulty_level}",
      "category": "General / Everyday",
      "context": "Essential adjective for expressing significance"
    }}
  ]
}}
"""


# --- File Processing Functions ---
def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file based on file type"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        if file_extension == 'txt':
            # Handle text files
            content = uploaded_file.read()
            if isinstance(content, bytes):
                return content.decode('utf-8')
            return str(content)
            
        elif file_extension == 'pdf':
            # Handle PDF files - improved error handling
            try:
                file_bytes = uploaded_file.read()
                if len(file_bytes) == 0:
                    st.error("PDF file appears to be empty")
                    return ""
                
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
                text = ""
                
                if len(pdf_reader.pages) == 0:
                    st.error("PDF has no pages")
                    return ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += page_text + "\n"
                    except Exception as page_error:
                        st.warning(f"Could not read page {page_num + 1}: {str(page_error)}")
                        continue
                
                if not text.strip():
                    st.error("Could not extract any readable text from PDF")
                    return ""
                    
                return text.strip()
                
            except Exception as pdf_error:
                st.error(f"Error reading PDF: {str(pdf_error)}")
                # Try alternative PDF processing if PyPDF2 fails
                try:
                    import pdfplumber
                    uploaded_file.seek(0)
                    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
                        text = ""
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        return text.strip()
                except ImportError:
                    st.error("Could not process PDF. Consider installing pdfplumber for better PDF support.")
                    return ""
                except Exception as plumber_error:
                    st.error(f"Alternative PDF processing also failed: {str(plumber_error)}")
                    return ""
            
        elif file_extension in ['docx', 'doc']:
            # Handle Word documents
            try:
                file_bytes = uploaded_file.read()
                if len(file_bytes) == 0:
                    st.error("Word document appears to be empty")
                    return ""
                
                doc = docx.Document(BytesIO(file_bytes))
                text = ""
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():  # Only add non-empty paragraphs
                        text += paragraph.text + "\n"
                
                if not text.strip():
                    st.error("Could not extract any readable text from Word document")
                    return ""
                    
                return text.strip()
                
            except Exception as docx_error:
                st.error(f"Error reading Word document: {str(docx_error)}")
                return ""
            
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
            
    except Exception as e:
        st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
        return ""


def show_file_upload_interface():
    """Show drag and drop file upload interface"""
    st.subheader("📁 Upload Spanish Documents")
    
    # File uploader with drag and drop
    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(SUPPORTED_FILE_TYPES)}"
    )
    
    if uploaded_files:
        st.success(f"📎 {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Show file details
        for i, file in enumerate(uploaded_files):
            with st.expander(f"📄 {file.name} ({file.size} bytes)"):
                file_text = extract_text_from_file(file)
                
                if file_text:
                    st.text_area(
                        f"Content preview for {file.name}",
                        value=file_text[:500] + ("..." if len(file_text) > 500 else ""),
                        height=150,
                        key=f"preview_{i}"
                    )
                    
                    # Store extracted text in session state for processing
                    if f"file_text_{i}" not in st.session_state:
                        st.session_state[f"file_text_{i}"] = file_text
                else:
                    st.error(f"Could not extract text from {file.name}")
        
        return uploaded_files
    
    return None


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
async def analyze_vocabulary_with_ai(text: str, provider_config: Dict, api_key: str = None,
                                     difficulty_level: str = "Intermediate") -> List[Dict]:
    """Use AI to intelligently extract and analyze vocabulary including phrases"""

    prompt = get_enhanced_prompt(text, difficulty_level)

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
                "max_tokens": 3000,
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

                        # Convert to our format matching Notion DB and filter out proper nouns
                        result = []
                        for item in vocabulary:
                            spanish_item = item.get("spanish", "").strip()
                            item_type = item.get("type", "Word")
                            
                            # For phrases, be less strict about proper noun filtering
                            if item_type == "Phrase" or not is_likely_proper_noun(spanish_item):
                                result.append({
                                    "Spanish": spanish_item,
                                    "English": item.get("english", ""),
                                    "Type": item_type,
                                    "Difficulty Level": item.get("difficulty", difficulty_level),
                                    "Category": item.get("category", "General / Everyday"),
                                    "Reveal Answer": False,
                                    "Correct Answer": "",  # Empty by default
                                    "Status": "Not started",
                                    "Date Added": datetime.today().date(),  # Use date object instead of string
                                    "Selected": True  # New field for selection
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
        "parent": {"database_id": database_id},
        "properties": {
            "Spanish": {
                "title": [{"text": {"content": row["Spanish"]}}]
            },
            "English": {
                "rich_text": [{"text": {"content": str(row["English"])}}]
            },
            "Date Added": {
                "date": {"start": today}
            }
        }
    }

    # Handle Type (select property) - NEW
    if "Type" in row and pd.notna(row["Type"]):
        properties["properties"]["Type"] = {
            "select": {"name": str(row["Type"])}
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


def push_to_notion(selected_df: pd.DataFrame, notion_token: str, database_id: str):
    """Push selected vocabulary to Notion with progress tracking"""
    try:
        # Get existing words in lowercase for comparison
        existing_words = {word.lower() for word in get_existing_words(notion_token, database_id)}

        # Clean and filter the DataFrame
        df_clean = selected_df.dropna(subset=["Spanish"])
        df_clean = df_clean[df_clean["Spanish"].str.strip() != ""]  # Remove empty strings

        # Check for duplicates (case-insensitive)
        df_clean["is_duplicate"] = df_clean["Spanish"].str.lower().isin(existing_words)
        duplicates = df_clean[df_clean["is_duplicate"]]
        new_words = df_clean[~df_clean["is_duplicate"]]

        if not duplicates.empty:
            st.warning(f"Found {len(duplicates)} duplicates that won't be added:")
            st.dataframe(duplicates[["Spanish", "English", "Type"]], use_container_width=True)

        if new_words.empty:
            st.warning("No new words to add after duplicate check")
            return False
        else:
            progress_bar = st.progress(0)
            success_count = 0

            for i, row in enumerate(new_words.to_dict("records")):
                if create_page(row, notion_token, database_id):
                    success_count += 1
                progress_bar.progress((i + 1) / len(new_words))
                time.sleep(0.3)  # Rate limiting

            st.success(f"🎉 Successfully added {success_count} new vocabulary items to Notion!")
            st.balloons()
            return True

    except Exception as e:
        st.error(f"Error pushing to Notion: {str(e)}")
        return False


def show_word_selection_interface():
    """Show interface for selecting/deselecting vocabulary words and phrases"""
    if 'vocabulary_df' not in st.session_state or st.session_state.vocabulary_df.empty:
        return

    df = st.session_state.vocabulary_df
    
    # Ensure Selected column exists
    if 'Selected' not in df.columns:
        df['Selected'] = True
        st.session_state.vocabulary_df = df

    st.subheader("🎯 Select Items to Add")
    
    # Quick action buttons
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("✅ Select All"):
            st.session_state.vocabulary_df['Selected'] = True
            st.rerun()
    
    with col2:
        if st.button("❌ Deselect All"):
            st.session_state.vocabulary_df['Selected'] = False
            st.rerun()
    
    with col3:
        if st.button("📝 Words Only"):
            st.session_state.vocabulary_df['Selected'] = False
            word_indices = df[df['Type'] == 'Word'].index
            st.session_state.vocabulary_df.loc[word_indices, 'Selected'] = True
            st.rerun()
    
    with col4:
        if st.button("💬 Phrases Only"):
            st.session_state.vocabulary_df['Selected'] = False
            phrase_indices = df[df['Type'] == 'Phrase'].index
            st.session_state.vocabulary_df.loc[phrase_indices, 'Selected'] = True
            st.rerun()
    
    with col5:
        if st.button("🎲 Random 10"):
            st.session_state.vocabulary_df['Selected'] = False
            random_indices = df.sample(n=min(10, len(df))).index
            st.session_state.vocabulary_df.loc[random_indices, 'Selected'] = True
            st.rerun()

    # Show selection stats
    selected_count = df['Selected'].sum()
    total_count = len(df)
    words_count = len(df[df['Type'] == 'Word'])
    phrases_count = len(df[df['Type'] == 'Phrase'])
    selected_words = len(df[(df['Selected'] == True) & (df['Type'] == 'Word')])
    selected_phrases = len(df[(df['Selected'] == True) & (df['Type'] == 'Phrase')])
    
    st.info(f"📊 Selected: {selected_count} / {total_count} items ({selected_words} words, {selected_phrases} phrases)")

    # Group by type for better organization
    words_df = df[df['Type'] == 'Word']
    phrases_df = df[df['Type'] == 'Phrase']

    # Show words section
    if not words_df.empty:
        with st.expander(f"📝 Words ({len(words_df)})", expanded=True):
            for idx, row in words_df.iterrows():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    current_selection = st.session_state.vocabulary_df.loc[idx, 'Selected']
                    new_selection = st.checkbox(
                        "Select",
                        value=current_selection,
                        key=f"select_word_{idx}",
                        label_visibility="collapsed"
                    )
                    
                    if new_selection != current_selection:
                        st.session_state.vocabulary_df.loc[idx, 'Selected'] = new_selection
                
                with col2:
                    # Color code based on selection
                    if st.session_state.vocabulary_df.loc[idx, 'Selected']:
                        st.markdown(f"**{row['Spanish']}** → *{row['English']}* | {row['Category']} | {row['Difficulty Level']}")
                    else:
                        st.markdown(f"~~{row['Spanish']} → {row['English']}~~ | {row['Category']} | {row['Difficulty Level']}")

    # Show phrases section
    if not phrases_df.empty:
        with st.expander(f"💬 Phrases ({len(phrases_df)})", expanded=True):
            for idx, row in phrases_df.iterrows():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    current_selection = st.session_state.vocabulary_df.loc[idx, 'Selected']
                    new_selection = st.checkbox(
                        "Select",
                        value=current_selection,
                        key=f"select_phrase_{idx}",
                        label_visibility="collapsed"
                    )
                    
                    if new_selection != current_selection:
                        st.session_state.vocabulary_df.loc[idx, 'Selected'] = new_selection
                
                with col2:
                    # Color code based on selection
                    if st.session_state.vocabulary_df.loc[idx, 'Selected']:
                        st.markdown(f"**{row['Spanish']}** → *{row['English']}* | {row['Category']} | {row['Difficulty Level']}")
                    else:
                        st.markdown(f"~~{row['Spanish']} → {row['English']}~~ | {row['Category']} | {row['Difficulty Level']}")


def show_review_section():
    """Enhanced review section with word and phrase selection"""
    if 'vocabulary_df' not in st.session_state or st.session_state.vocabulary_df.empty:
        return

    st.divider()
    st.header("📋 Review & Select Vocabulary")

    df = st.session_state.vocabulary_df
    
    # Ensure Selected column exists
    if 'Selected' not in df.columns:
        df['Selected'] = True
        st.session_state.vocabulary_df = df

    # Show word selection interface
    show_word_selection_interface()

    # Show preview of selected words
    selected_df = df[df['Selected'] == True].copy()
    
    if not selected_df.empty:
        st.subheader("📄 Preview: Selected Items")
        
        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Selected", len(selected_df))
        with col2:
            words_selected = len(selected_df[selected_df['Type'] == 'Word'])
            st.metric("Words", words_selected)
        with col3:
            phrases_selected = len(selected_df[selected_df['Type'] == 'Phrase'])
            st.metric("Phrases", phrases_selected)
        with col4:
            if 'Difficulty Level' in selected_df.columns:
                intermediate_count = len(selected_df[selected_df['Difficulty Level'] == 'Intermediate'])
                st.metric("Intermediate", intermediate_count)

        # Show breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Difficulty Level' in selected_df.columns:
                difficulty_counts = selected_df['Difficulty Level'].value_counts()
                st.write("**📊 Difficulty Breakdown:**")
                for diff, count in difficulty_counts.items():
                    st.write(f"- {diff}: {count} items")

        with col2:
            if 'Type' in selected_df.columns:
                type_counts = selected_df['Type'].value_counts()
                st.write("**🏷️ Type Breakdown:**")
                for item_type, count in type_counts.items():
                    st.write(f"- {item_type}: {count} items")

        # Show the selected data in a clean format
        display_df = selected_df[['Spanish', 'English', 'Type', 'Difficulty Level', 'Category']].copy()
        st.dataframe(display_df, use_container_width=True)

        # Push to Notion button
        secrets = get_secrets()
        if st.button("🚀 Push Selected Items to Notion", type="primary"):
            if not secrets.get("NOTION_TOKEN") or not secrets.get("DATABASE_ID"):
                st.warning("Please configure Notion connection in the sidebar")
            else:
                with st.spinner("Pushing selected items to Notion..."):
                    success = push_to_notion(selected_df, secrets["NOTION_TOKEN"], secrets["DATABASE_ID"])
                    
                    if success:
                        # Clear the vocabulary after successful push
                        if st.button("🗑️ Clear Vocabulary List"):
                            empty_data = {col: [] for col in DEFAULT_FIELDS + ['Selected']}
                            st.session_state.vocabulary_df = pd.DataFrame(empty_data)
                            st.rerun()
    else:
        st.warning("⚠️ No items selected. Please select at least one word or phrase to push to Notion.")


# --- Enhanced Web Scraping Functions ---
def setup_chrome_driver():
    """Setup Chrome driver with optimal settings"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-javascript")  # We'll enable selectively
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        st.error(f"Failed to setup Chrome driver: {e}")
        return None


async def fetch_article_text(url: str) -> str:
    """Enhanced article fetching with multiple fallback methods"""
    st.info("🔍 Starting article extraction...")

    # Method 1: Static scraping with custom headers
    text = await _fetch_static(url)
    if len(text) > 300:
        st.success("✅ Successfully extracted using static method")
        return text

    # Method 2: Selenium with JavaScript enabled
    text = await _fetch_with_selenium(url)
    if len(text) > 300:
        st.success("✅ Successfully extracted using Selenium")
        return text

    # Method 3: Try different user agents
    text = await _fetch_with_different_headers(url)
    if len(text) > 300:
        st.success("✅ Successfully extracted with alternative headers")
        return text

    st.error("❌ All extraction methods failed")
    return ""


async def _fetch_static(url: str) -> str:
    """Enhanced static scraping with better headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    }

    try:
        async with httpx.AsyncClient(
                headers=headers,
                timeout=15.0,
                follow_redirects=True,
                verify=ssl_context
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()

            # Try to find main content areas
            content_selectors = [
                'article', '[role="main"]', 'main', '.content', '.post-content',
                '.entry-content', '.article-content', '.post-body', '.text-content'
            ]

            text = ""
            for selector in content_selectors:
                content_area = soup.select_one(selector)
                if content_area:
                    text = content_area.get_text()
                    break

            # Fallback to body if no specific content area found
            if not text:
                text = soup.get_text()

            # Clean up text
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    except Exception as e:
        st.warning(f"Static fetch failed: {str(e)}")
        return ""


async def _fetch_with_selenium(url: str) -> str:
    """Selenium-based scraping for JavaScript-heavy sites"""
    try:
        # Use asyncio to run selenium in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _selenium_scrape, url)
        return text
    except Exception as e:
        st.warning(f"Selenium fetch failed: {str(e)}")
        return ""


def _selenium_scrape(url: str) -> str:
    """Selenium scraping function (runs in thread pool)"""
    driver = None
    try:
        driver = setup_chrome_driver()
        if not driver:
            return ""

        # Set page load timeout
        driver.set_page_load_timeout(15)
        driver.implicitly_wait(10)

        # Navigate to page
        driver.get(url)

        # Wait for content to load
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "article"))
            )
        except:
            # If no article tag, wait for body
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

        # Try to find main content
        content_selectors = [
            "article", "[role='main']", "main", ".content", ".post-content",
            ".entry-content", ".article-content", ".post-body"
        ]

        text = ""
        for selector in content_selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                text = element.text
                if len(text) > 200:
                    break
            except:
                continue

        # Fallback to body text
        if not text:
            text = driver.find_element(By.TAG_NAME, "body").text

        return re.sub(r'\s+', ' ', text).strip()

    except Exception as e:
        st.warning(f"Selenium execution failed: {str(e)}")
        return ""
    finally:
        if driver:
            driver.quit()


async def _fetch_with_different_headers(url: str) -> str:
    """Try different user agents and headers"""
    user_agents = [
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]

    for user_agent in user_agents:
        try:
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'es,en-US;q=0.7,en;q=0.3',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }

            async with httpx.AsyncClient(
                    headers=headers,
                    timeout=10.0,
                    follow_redirects=True
            ) as client:
                response = await client.get(url)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer']):
                        element.decompose()

                    text = soup.get_text()
                    text = re.sub(r'\s+', ' ', text).strip()

                    if len(text) > 300:
                        return text

        except Exception:
            continue

    return ""


async def process_article(article_url: str, provider_config: Dict, api_key: str = None,
                          difficulty_level: str = "Intermediate"):
    with st.spinner("Fetching article..."):
        article_text = await fetch_article_text(article_url)

        if not article_text:
            st.error(
                "❌ Could not fetch article text. The site might be blocking automated access or requires JavaScript.")
            st.info("💡 Try copying and pasting the text manually in the 'Manual Entry' tab or upload a file.")
            return

        if len(article_text) < 200:
            st.warning("⚠️ Article text seems too short. Please check the URL or try a different article.")
            st.info(f"Extracted text length: {len(article_text)} characters")
            return

        st.session_state.article_text = article_text

        # Show preview of article
        with st.expander("📄 Article Preview"):
            st.text_area("Extracted Article Text",
                         value=article_text[:1500] + ("..." if len(article_text) > 1500 else ""),
                         height=300,
                         help=f"Total characters: {len(article_text)}")

    with st.spinner("🤖 AI is analyzing vocabulary and phrases..."):
        vocabulary = await analyze_vocabulary_with_ai(article_text, provider_config, api_key, difficulty_level)

        if vocabulary:
            st.session_state.vocabulary_df = pd.DataFrame(vocabulary)
            
            words_count = len([v for v in vocabulary if v.get('Type') == 'Word'])
            phrases_count = len([v for v in vocabulary if v.get('Type') == 'Phrase'])
            
            st.success(f"🎉 AI extracted {len(vocabulary)} items ({words_count} words, {phrases_count} phrases) at {difficulty_level} level!")

            # Show type breakdown
            if vocabulary:
                type_counts = pd.Series([v.get('Type', 'Word') for v in vocabulary]).value_counts()
                st.write("**📊 Type Breakdown:**")
                for item_type, count in type_counts.items():
                    st.write(f"- {item_type}: {count} items")

            # Show difficulty breakdown
            if vocabulary and 'Difficulty Level' in vocabulary[0]:
                difficulty_counts = pd.Series([v['Difficulty Level'] for v in vocabulary]).value_counts()
                st.write("**📊 Difficulty Breakdown:**")
                for diff, count in difficulty_counts.items():
                    st.write(f"- {diff}: {count} items")

            # Show category breakdown
            if vocabulary and 'Category' in vocabulary[0]:
                category_counts = pd.Series([v['Category'] for v in vocabulary]).value_counts()
                st.write("**🏷️ Category Breakdown:**")
                for cat, count in category_counts.items():
                    st.write(f"- {cat}: {count} items")
        else:
            st.warning(
                "⚠️ AI could not extract vocabulary. Please try a different article or check your API configuration.")


# --- Additional Helper Function for Direct Text Processing ---
async def process_text_directly(text: str, provider_config: Dict, api_key: str = None,
                                difficulty_level: str = "Intermediate"):
    """Process text directly without URL fetching"""

    with st.spinner("🤖 AI is analyzing vocabulary and phrases..."):
        vocabulary = await analyze_vocabulary_with_ai(text, provider_config, api_key, difficulty_level)

        if vocabulary:
            st.session_state.vocabulary_df = pd.DataFrame(vocabulary)
            
            words_count = len([v for v in vocabulary if v.get('Type') == 'Word'])
            phrases_count = len([v for v in vocabulary if v.get('Type') == 'Phrase'])
            
            st.success(f"🎉 AI extracted {len(vocabulary)} items ({words_count} words, {phrases_count} phrases) at {difficulty_level} level!")

            # Show text preview
            with st.expander("📄 Text Preview"):
                st.text_area("Analyzed Text",
                             value=text[:1500] + ("..." if len(text) > 1500 else ""),
                             height=200,
                             help=f"Total characters: {len(text)}")

            # Show type breakdown
            if vocabulary:
                type_counts = pd.Series([v.get('Type', 'Word') for v in vocabulary]).value_counts()
                st.write("**📊 Type Breakdown:**")
                for item_type, count in type_counts.items():
                    st.write(f"- {item_type}: {count} items")

            # Show difficulty breakdown
            if vocabulary and 'Difficulty Level' in vocabulary[0]:
                difficulty_counts = pd.Series([v['Difficulty Level'] for v in vocabulary]).value_counts()
                st.write("**📊 Difficulty Breakdown:**")
                for diff, count in difficulty_counts.items():
                    st.write(f"- {diff}: {count} items")

            # Show category breakdown
            if vocabulary and 'Category' in vocabulary[0]:
                category_counts = pd.Series([v['Category'] for v in vocabulary]).value_counts()
                st.write("**🏷️ Category Breakdown:**")
                for cat, count in category_counts.items():
                    st.write(f"- {cat}: {count} items")
        else:
            st.warning("⚠️ AI could not extract vocabulary. Please try different text or check your API configuration.")


async def process_uploaded_files(uploaded_files, provider_config: Dict, api_key: str = None,
                                difficulty_level: str = "Intermediate"):
    """Process multiple uploaded files and extract vocabulary"""
    
    all_text = ""
    file_info = []
    
    for i, file in enumerate(uploaded_files):
        file_text = extract_text_from_file(file)
        if file_text:
            all_text += f"\n\n--- {file.name} ---\n{file_text}"
            file_info.append(f"✅ {file.name}: {len(file_text)} characters")
        else:
            file_info.append(f"❌ {file.name}: Failed to extract text")
    
    # Show file processing results
    st.info("📁 File Processing Results:")
    for info in file_info:
        st.write(info)
    
    if not all_text.strip():
        st.error("❌ No text could be extracted from any uploaded files.")
        return
    
    st.info(f"📊 Total text extracted: {len(all_text)} characters from {len(uploaded_files)} files")
    
    # Process the combined text
    await process_text_directly(all_text, provider_config, api_key, difficulty_level)


# --- Streamlit App ---
def main():
    st.title("🎓 AI-Powered Spanish Vocabulary Collector")
    st.markdown("*Intelligently extract useful vocabulary and phrases from Spanish content using AI*")

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

    # Main tabs - Updated with file upload tab
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Files", "📰 From Article URL", "✏️ Manual Entry", "📝 From Text"])

    with tab1:
        st.header("📁 Upload Spanish Documents")
        st.info("💡 **New!** Drag and drop files or click to browse. Supports TXT, PDF, DOCX, and DOC files.")
        
        uploaded_files = show_file_upload_interface()
        
        if uploaded_files:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("🤖 Extract from Files", type="primary"):
                    if provider_config["requires_key"] and not ai_api_key:
                        st.warning(f"Please configure {selected_provider} API key")
                    else:
                        difficulty_level = get_difficulty_from_slider(difficulty_slider)
                        asyncio.run(process_uploaded_files(uploaded_files, provider_config, ai_api_key, difficulty_level))

    with tab2:
        st.header("Extract Vocabulary from Spanish Article")

        st.info(
            "💡 **Tip**: This tool works best with news articles, blogs, and educational content. Some sites may block automated access.")

        article_url = st.text_input("🔗 Enter Spanish Article URL",
                                    placeholder="https://elpais.com/...",
                                    help="Paste any Spanish article URL here")

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

    with tab3:
        st.header("Manually Add Vocabulary")

        # Initialize DataFrame with proper columns and types if it doesn't exist
        if 'vocabulary_df' not in st.session_state:
            st.session_state.vocabulary_df = pd.DataFrame({
                "Spanish": pd.Series(dtype='str'),
                "English": pd.Series(dtype='str'),
                "Type": pd.Series(dtype='str'),
                "Difficulty Level": pd.Series(dtype='str'),
                "Category": pd.Series(dtype='str'),
                "Reveal Answer": pd.Series(dtype='bool'),
                "Correct Answer": pd.Series(dtype='str'),
                "Status": pd.Series(dtype='str'),
                "Date Added": pd.Series(dtype='datetime64[ns]'),
                "Selected": pd.Series(dtype='bool')
            })

        # Ensure all columns exist and have correct types
        df = st.session_state.vocabulary_df
        for col in DEFAULT_FIELDS + ['Selected']:
            if col not in df.columns:
                if col == "Reveal Answer" or col == "Selected":
                    df[col] = False
                elif col == "Date Added":
                    df[col] = pd.to_datetime(datetime.today().date())
                elif col == "Type":
                    df[col] = "Word"
                else:
                    df[col] = ""

        # Convert date column to datetime if it's not already
        if 'Date Added' in df:
            df['Date Added'] = pd.to_datetime(df['Date Added'])

        # Create the data editor with robust column configuration
        edited_df = st.data_editor(
            df.drop(columns=['Selected']) if 'Selected' in df.columns else df,  # Hide Selected column from manual editor
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
                "Type": st.column_config.SelectboxColumn(
                    "Type",
                    options=TYPE_OPTIONS,
                    default="Word"
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
            # Add Selected column back with all True for manually entered words
            edited_df['Selected'] = True
            st.session_state.vocabulary_df = edited_df
            st.success("Vocabulary list updated!")

    with tab4:
        st.header("Extract Vocabulary from Text")
        st.info("💡 **Perfect for when URLs don't work!** Copy and paste Spanish text directly here.")

        # Text input area
        spanish_text = st.text_area(
            "📝 Paste Spanish Text Here",
            height=300,
            placeholder="Paste your Spanish article, blog post, or any text here...",
            help="Copy text from any Spanish source and paste it here for AI analysis"
        )

        # Character count
        if spanish_text:
            char_count = len(spanish_text)
            st.caption(f"Characters: {char_count}")

            if char_count < 100:
                st.warning("⚠️ Text seems short. For best results, use at least 100 characters.")
            elif char_count > 5000:
                st.info("ℹ️ Long text detected. AI will analyze the first 4000 characters.")

        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("🤖 Analyze Text with AI", type="primary"):
                if not spanish_text.strip():
                    st.warning("Please paste some Spanish text")
                elif len(spanish_text.strip()) < 50:
                    st.warning("Please provide more text (at least 50 characters)")
                elif provider_config["requires_key"] and not ai_api_key:
                    st.warning(f"Please configure {selected_provider} API key")
                else:
                    difficulty_level = get_difficulty_from_slider(difficulty_slider)
                    asyncio.run(process_text_directly(spanish_text, provider_config, ai_api_key, difficulty_level))

    # Enhanced Review and Push Section
    show_review_section()


if __name__ == "__main__":
    main()
