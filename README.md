# 🎓 AI-Powered Spanish Vocabulary Collector

An intelligent Streamlit web application that automatically extracts and analyzes useful Spanish vocabulary from articles and text using AI, then seamlessly integrates with Notion databases for vocabulary management.

## 🌟 What It Does

This application revolutionizes Spanish language learning by:

- **Intelligently extracting vocabulary** from Spanish articles, blogs, and texts using advanced AI analysis
- **Automatically categorizing words** by difficulty level (Beginner/Intermediate/Advanced) and topic categories
- **Filtering out common words** to focus on educationally valuable vocabulary
- **Seamlessly syncing** with your Notion database for organized vocabulary management
- **Providing multiple input methods** - URL scraping, direct text input, or manual entry

## 🚀 Key Features

### 🤖 AI-Powered Analysis
- Uses multiple AI providers (OpenRouter, Groq, or local Ollama)
- Extracts 15-25 most useful vocabulary words per analysis
- Intelligent difficulty assessment and categorization
- Context-aware word selection prioritizing educational value

### 🌐 Advanced Web Scraping
- Multi-method article extraction with fallback strategies
- Static scraping with optimized headers
- Selenium-based scraping for JavaScript-heavy sites
- Support for most Spanish news sites and blogs

### 📚 Smart Categorization
Words are automatically categorized into:
- Culture / Media
- Health / Body
- Politics / Economy
- Emotions / Relationships
- Travel / Tourism
- Academic / Education
- Technology
- Professional / Business
- General / Everyday
- Useful Phrases

### 🔄 Notion Integration
- Automatic duplicate detection
- Seamless database synchronization
- Configurable field mapping
- Batch processing with progress tracking

## 🛠️ How It's Built

### Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **Web Scraping**: 
  - `requests` + `BeautifulSoup4` for static content
  - `Selenium` + `ChromeDriver` for dynamic content
  - `httpx` for async HTTP operations
- **AI Integration**: Multiple provider support (OpenAI-compatible APIs)
- **Database**: Notion API integration
- **Data Processing**: `pandas` for data manipulation

### Architecture
```
User Input (URL/Text) 
    ↓
Multi-Method Web Scraping
    ↓
AI Vocabulary Analysis
    ↓
Data Processing & Categorization
    ↓
Duplicate Detection
    ↓
Notion Database Sync
```

### AI Analysis Process
1. **Text Preprocessing**: Clean and prepare extracted content
2. **Intelligent Selection**: AI analyzes text for educationally valuable vocabulary
3. **Contextual Categorization**: Automatically assigns categories and difficulty levels
4. **Quality Filtering**: Removes overly basic words and proper nouns
5. **Structured Output**: Returns organized vocabulary data ready for database insertion

## 💡 Why It's Useful

### For Spanish Learners
- **Saves hours** of manual vocabulary extraction and organization
- **Focuses on useful words** rather than every word in an article
- **Provides context** for why each word is important
- **Organizes learning** with automatic difficulty and category classification
- **Prevents duplicates** in your study materials

### For Educators
- **Quickly generates vocabulary lists** from authentic Spanish content
- **Ensures appropriate difficulty levels** for different student groups
- **Creates organized study materials** automatically
- **Tracks vocabulary coverage** across different topics

### For Language Enthusiasts
- **Discovers new vocabulary** from current events and trending topics
- **Maintains organized vocabulary database** for long-term learning
- **Analyzes learning progress** through categorized vocabulary tracking

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Notion account with API access
- AI API key (OpenRouter, Groq, or local Ollama setup)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spanish-vocab-collector.git
   cd spanish-vocab-collector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional - can also use Streamlit interface)
   ```bash
   # Create .streamlit/secrets.toml
   NOTION_TOKEN = "your_notion_integration_token"
   DATABASE_ID = "your_notion_database_id"
   GROQ_TOKEN = "your_groq_api_key"  # Optional
   OPENROUTER_TOKEN = "your_openrouter_api_key"  # Optional
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

### Notion Setup

1. **Create a Notion Integration**
   - Go to [Notion Developers](https://developers.notion.com/)
   - Create a new integration and copy the token

2. **Create a Database**
   - Create a new database in Notion with these properties:
     - `Spanish` (Title)
     - `English` (Text)
     - `Difficulty Level` (Select: Beginner, Intermediate, Advanced)
     - `Category` (Select: Culture/Media, Health/Body, etc.)
     - `Reveal Answer` (Checkbox)
     - `Correct Answer` (Text)
     - `Status` (Status: Not started, Done)
     - `Date Added` (Date)

3. **Share Database with Integration**
   - Share your database with your integration

### AI Provider Setup

Choose one of these options:

**Option 1: OpenRouter (Recommended for beginners)**
- Sign up at [OpenRouter](https://openrouter.ai/)
- Get free credits and API key
- Use model: `meta-llama/llama-3.1-8b-instruct:free`

**Option 2: Groq (Fast and free)**
- Sign up at [Groq Console](https://console.groq.com/)
- Get free API key
- Use model: `llama3-8b-8192`

**Option 3: Local Ollama (Advanced users)**
- Install [Ollama](https://ollama.ai/)
- Run `ollama pull llama3.1`
- Use local endpoint

## 📱 Usage Guide

### Method 1: Article URL
1. Set your difficulty preference (1-10 slider)
2. Configure Notion and AI credentials
3. Paste a Spanish article URL
4. Click "Extract with AI"
5. Review extracted vocabulary
6. Push to Notion database

### Method 2: Direct Text Input
1. Copy Spanish text from any source
2. Paste into the text area
3. Click "Analyze Text with AI"
4. Review and edit results
5. Push to Notion

### Method 3: Manual Entry
1. Use the data editor to manually add vocabulary
2. Set categories and difficulty levels
3. Push to Notion when ready

## 🎯 Best Practices

### For Optimal Results
- **Use articles with 200+ words** for better AI analysis
- **Choose appropriate difficulty levels** based on your learning goals
- **Review AI suggestions** before pushing to Notion
- **Test different AI providers** to find what works best for your content

### Recommended Sources
- Spanish news sites (El País, BBC Mundo, CNN Español)
- Educational blogs and websites
- Government and institutional websites
- Academic articles and papers

## 🔧 Configuration Options

### Difficulty Levels
- **Slider 1-3**: Beginner (basic everyday vocabulary)
- **Slider 4-7**: Intermediate (more complex and specific terms)
- **Slider 8-10**: Advanced (technical, academic, or specialized vocabulary)

### AI Provider Comparison
| Provider | Speed | Cost | Setup Difficulty | Best For |
|----------|-------|------|------------------|----------|
| OpenRouter | Medium | Free credits | Easy | Beginners |
| Groq | Fast | Free | Easy | Speed-focused users |
| Ollama | Fast | Free | Advanced | Privacy-conscious users |

## 🛡️ Privacy & Security

- **API keys** are stored securely in Streamlit secrets
- **No data retention** - articles are processed in memory only
- **Notion integration** uses official API with proper authentication
- **Optional local processing** with Ollama for complete privacy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- AI providers: [OpenRouter](https://openrouter.ai/), [Groq](https://groq.com/), [Ollama](https://ollama.ai/)
- [Notion API](https://developers.notion.com/) for database integration
- Web scraping powered by Selenium and BeautifulSoup

## 📞 Support

If you encounter any issues or have questions:
1. Check the **Test Connections** feature in the sidebar
2. Verify your API credentials and Notion database setup
3. Try different AI providers or articles
4. Open an issue on GitHub for technical problems

---

**Happy Learning! 🎉** Transform your Spanish vocabulary acquisition with AI-powered intelligence!
