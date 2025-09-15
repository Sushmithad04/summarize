

import streamlit as st
from transformers import pipeline

# Load Hugging Face models (summarization and sentiment)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

summarizer = load_summarizer()
sentiment_analyzer = load_sentiment()

st.title("📝 E-Consultation Comments Dashboard")
st.write("Paste multiple comments (one per line) to get summaries and sentiment analysis:")

user_input = st.text_area("Paste comments here (one per line)", height=200)

if st.button("Analyze & Summarize"):
    if user_input.strip():
        comments = [c.strip() for c in user_input.strip().split('\n') if c.strip()]
        if not comments:
            st.warning("Please enter at least one comment.")
        else:
            with st.spinner("Processing comments..."):
                results = []
                all_text = " ".join(comments)
                # Summarize all comments together
                overall_summary = summarizer(all_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                for comment in comments:
                    # Summarize each comment (if long enough)
                    if len(comment.split()) > 15:
                        summary = summarizer(comment, max_length=60, min_length=15, do_sample=False)[0]['summary_text']
                    else:
                        summary = comment  # Too short to summarize
                    sentiment = sentiment_analyzer(comment)[0]
                    results.append({
                        "Comment": comment,
                        "Summary": summary,
                        "Sentiment": sentiment['label'],
                        "Score": round(sentiment['score'], 2)
                    })
            st.subheader("Overall Summary of All Comments:")
            st.info(overall_summary)
            st.subheader("Detailed Comment Analysis:")
            st.dataframe(results)
    else:
        st.warning("Please enter some comments to analyze.")
