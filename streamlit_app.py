# Update the CATEGORY_OPTIONS to include 'Useful Phrases'
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
    "Useful Phrases"  # Added new category
]

# Add this function to map slider values to difficulty levels
def get_difficulty_from_slider(slider_value: int) -> str:
    """Map slider value (1-10) to difficulty level"""
    if slider_value <= 3:
        return "Beginner"
    elif 4 <= slider_value <= 7:
        return "Intermediate"
    else:
        return "Advanced"

# In the main function, add the difficulty slider in the sidebar
def main():
    # ... (existing code until sidebar)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Add difficulty level slider
        st.subheader("Difficulty Level")
        difficulty_slider = st.slider(
            "Select difficulty range (1-10)",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="1-3: Beginner, 4-7: Intermediate, 8-10: Advanced"
        )
        
        # Display visual indicator
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Level", difficulty_slider)
        with col2:
            difficulty_level = get_difficulty_from_slider(difficulty_slider)
            st.metric("Difficulty", difficulty_level)
        with col3:
            # Simple icon display based on level
            if difficulty_slider <= 3:
                st.write("🌱 Beginner")
            elif difficulty_slider <= 7:
                st.write("🌿 Intermediate")
            else:
                st.write("🎓 Advanced")
                
        # ... (rest of existing sidebar code)

# Modify the process_article function to filter by difficulty
async def process_article(article_url: str, provider_config: Dict, api_key: str = None):
    # ... (existing code until vocabulary extraction)
    
    if vocabulary:
        # Convert to DataFrame
        df = pd.DataFrame(vocabulary)
        
        # Filter based on difficulty level
        difficulty_mapping = {
            "Beginner": [1, 2, 3],
            "Intermediate": [4, 5, 6, 7],
            "Advanced": [8, 9, 10]
        }
        
        selected_difficulty = get_difficulty_from_slider(st.session_state.get('difficulty_slider', 5))
        df = df[df['Difficulty Level'] == selected_difficulty]
        
        if df.empty:
            st.warning(f"No words found at {selected_difficulty} level. Try adjusting the difficulty slider.")
            return
            
        st.session_state.vocabulary_df = df
        st.success(f"AI extracted {len(df)} {selected_difficulty} vocabulary items!")
        
        # ... (rest of existing code)

# Modify the push to Notion section to check for duplicates
if 'vocabulary_df' in st.session_state and not st.session_state.vocabulary_df.empty:
    # ... (existing code until push button)
    
    if st.button("🚀 Push to Notion", type="primary"):
        if not notion_token or not database_id:
            st.warning("Please configure Notion connection")
        else:
            with st.spinner("Pushing to Notion..."):
                try:
                    # Get existing words in lowercase for comparison
                    existing_words = {word.lower() for word in get_existing_words(notion_token, database_id)}
                    
                    # Clean and filter the DataFrame
                    df_clean = st.session_state.vocabulary_df.dropna(subset=["Spanish"])
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
                        return
                    
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
