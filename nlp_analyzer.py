import streamlit as st
import pandas as pd

@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning("spaCy model not found. Attempting to download...")
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy: {e}")
        return None

def analyze_sentiment(text):
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        if polarity > 0.1:
            classification = "Positive"
            emoji = "😊"
            color = "#22c55e"
        elif polarity < -0.1:
            classification = "Negative"
            emoji = "😞"
            color = "#ef4444"
        else:
            classification = "Neutral"
            emoji = "😐"
            color = "#6b7280"

        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'classification': classification,
            'emoji': emoji,
            'color': color
        }
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return None

def extract_entities(text, nlp):
    try:
        doc = nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                'Text': ent.text,
                'Label': ent.label_,
                'Description': spacy.explain(ent.label_) or ent.label_
            })

        return entities, doc
    except Exception as e:
        st.error(f"Error extracting entities: {e}")
        return [], None

def render_nlp_analyzer():
    st.header("💬 NLP Sentiment & Entity Analysis")
    st.markdown("Analyze text using spaCy for Named Entity Recognition and TextBlob for sentiment analysis.")

    example_reviews = {
        "Positive Review": "This product is absolutely amazing! Best purchase I've ever made. The quality exceeded my expectations and the customer service was fantastic. Highly recommend to everyone!",
        "Negative Review": "Terrible experience. The product broke after two days and customer support was unhelpful. Complete waste of money. I'm very disappointed and frustrated.",
        "Neutral Review": "The product arrived on time. It works as described in the manual. Nothing special, but it does the job. The price seems fair for what you get.",
        "Mixed Review": "I bought this from Amazon last week. The delivery was fast and the packaging was great. However, the product quality is mediocre. Some features work well, others don't. Not sure if I would recommend it."
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Text")

        selected_example = st.selectbox(
            "Try an example:",
            ["Custom Input"] + list(example_reviews.keys())
        )

        if selected_example != "Custom Input":
            default_text = example_reviews[selected_example]
        else:
            default_text = ""

        user_text = st.text_area(
            "Enter text for analysis:",
            value=default_text,
            height=150,
            placeholder="Enter a review, comment, or any text you'd like to analyze..."
        )

        analyze_button = st.button("🔍 Analyze Text", use_container_width=True, type="primary")

    with col2:
        st.subheader("Quick Stats")
        if user_text:
            word_count = len(user_text.split())
            char_count = len(user_text)
            sentence_count = user_text.count('.') + user_text.count('!') + user_text.count('?')

            st.metric("Words", word_count)
            st.metric("Characters", char_count)
            st.metric("Sentences", max(1, sentence_count))

    if analyze_button and user_text:
        with st.spinner("Analyzing text..."):
            st.markdown("---")

            sentiment = analyze_sentiment(user_text)

            if sentiment:
                st.subheader(f"📊 Sentiment Analysis {sentiment['emoji']}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background-color: {sentiment['color']}22; border-radius: 10px; border: 2px solid {sentiment['color']}'>
                        <h2 style='color: {sentiment['color']}'>{sentiment['emoji']}</h2>
                        <h3>{sentiment['classification']}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    polarity_pct = (sentiment['polarity'] + 1) * 50
                    st.metric(
                        "Polarity Score",
                        f"{sentiment['polarity']:.3f}",
                        help="Range: -1 (negative) to +1 (positive)"
                    )
                    st.progress(polarity_pct / 100)

                with col3:
                    st.metric(
                        "Subjectivity Score",
                        f"{sentiment['subjectivity']:.3f}",
                        help="Range: 0 (objective) to 1 (subjective)"
                    )
                    st.progress(sentiment['subjectivity'])

                with st.expander("ℹ️ Understanding Sentiment Scores"):
                    st.markdown("""
                    **Polarity:**
                    - **Positive** (> 0.1): Indicates positive sentiment
                    - **Neutral** (-0.1 to 0.1): Indicates neutral sentiment
                    - **Negative** (< -0.1): Indicates negative sentiment

                    **Subjectivity:**
                    - **Subjective** (> 0.5): Personal opinions, emotions
                    - **Objective** (< 0.5): Factual information
                    """)

            st.markdown("---")
            st.subheader("🏷️ Named Entity Recognition (NER)")

            nlp = load_spacy_model()

            if nlp:
                entities, doc = extract_entities(user_text, nlp)

                if entities:
                    st.success(f"Found {len(entities)} named entities in the text.")

                    df = pd.DataFrame(entities)
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    entity_counts = df['Label'].value_counts()

                    if len(entity_counts) > 0:
                        st.markdown("**Entity Distribution:**")

                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(8, 4))
                        entity_counts.plot(kind='barh', ax=ax, color='#3b82f6')
                        ax.set_xlabel('Count')
                        ax.set_ylabel('Entity Type')
                        ax.set_title('Named Entity Distribution')
                        plt.tight_layout()
                        st.pyplot(fig)

                    with st.expander("📝 View Annotated Text"):
                        try:
                            import spacy
                            from spacy import displacy

                            html = displacy.render(doc, style="ent", jupyter=False)
                            st.markdown(html, unsafe_allow_html=True)
                        except:
                            st.info("Text annotation display not available.")

                else:
                    st.info("No named entities detected in the text.")

                with st.expander("ℹ️ Common Entity Types"):
                    st.markdown("""
                    - **PERSON**: People, including fictional characters
                    - **ORG**: Organizations, companies, agencies
                    - **GPE**: Geo-Political Entities (countries, cities, states)
                    - **DATE**: Absolute or relative dates or periods
                    - **MONEY**: Monetary values, including unit
                    - **PRODUCT**: Objects, vehicles, foods, etc.
                    - **EVENT**: Named events (sports, wars, etc.)
                    - **LOC**: Non-GPE locations (mountains, water bodies)
                    """)
    elif analyze_button:
        st.warning("Please enter some text to analyze.")

    with st.expander("ℹ️ About NLP Analysis"):
        st.markdown("""
        **spaCy:**
        - Industrial-strength Natural Language Processing library
        - Performs Named Entity Recognition (NER) to identify and classify entities
        - Uses pre-trained models for English language understanding

        **TextBlob:**
        - Simple NLP library built on NLTK
        - Performs sentiment analysis using lexicon-based approach
        - Returns polarity (negative to positive) and subjectivity (objective to subjective)

        **Use Cases:**
        - Customer review analysis
        - Social media monitoring
        - Content categorization
        - Brand sentiment tracking
        """)
