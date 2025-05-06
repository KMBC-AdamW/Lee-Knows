import streamlit as st
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from time import sleep
from datetime import datetime
import csv

# Set page configuration
st.set_page_config(page_title="Lee Knows – The NEW One Stop Shop", layout="wide")

# Load metadata from a JSON file
with open("faiss_index_metadata.json") as f:
    metadata = json.load(f)

# Extract all unique departments from the metadata
all_departments = sorted(set(d.get("department", "Unknown") for d in metadata if d.get("department")))

# Load the embedding model and FAISS index
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_index.index")
    return model, index

model, index = load_model_and_index()

# Cache the embedding of a given text query
@st.cache_data(show_spinner=False)
def get_embedding(text):
    return model.encode([text])

# Initialize last_query only if not set
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Header with avatar and title
col1, col2 = st.columns([1, 6])
with col1:
    st.image("lee_avatar.png", width=220, output_format="auto")
with col2:
    st.title("Lee Knows – The NEW One Stop Shop")

# Display Knowsley Council logo in the sidebar
st.sidebar.image("https://www.knowsley.gov.uk/themes/custom/knowsley/logo.svg", width=150)

# Department filters
st.sidebar.header("Filter your results")
selected_departments = st.sidebar.multiselect(
    "Select departments:",
    options=all_departments,
    default=[],
    help="Choose one or more departments to filter Lee's responses"
)

if not selected_departments:
    selected_departments = all_departments

# Initialize session state variables
if "chat" not in st.session_state:
    st.session_state.chat = []
if "log" not in st.session_state:
    st.session_state.log = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
    st.session_state.log = []

# Search top relevant chunks using FAISS and rerank based on relevance
def search_faiss(query_text):
    # Generate embedding for the query
    embedding = get_embedding(query_text)
    D, I = index.search(np.array(embedding).astype("float32"), k=30)

    # Collect candidate documents
    initial_results = []
    for i in I[0]:
        if 0 <= i < len(metadata):
            doc = metadata[i]
            initial_results.append(doc)

    # Define stopwords and extract meaningful keywords from query
    stopwords = {"how", "do", "i", "can", "a", "the", "is", "to", "of", "and", "you", "your", "for"}
    q_keywords = [word for word in query_text.lower().split() if word not in stopwords]

    # Scoring function to prioritise relevant, link-rich chunks
    def score(doc):
        title = doc.get('title', '').lower()
        url = doc.get('url', '').lower()
        department = doc.get('department', '').lower()
        body = doc.get('text', '').lower()
        links = doc.get('links', [])
        has_important_link = any(link.get("important") for link in links)

        score = 0
        if any(word in title for word in q_keywords):
            score += 4  # boosted title match
        if any(word in url for word in q_keywords):
            score += 1
        if any(word in body for word in q_keywords):
            score += 3
        if query_text.lower() in title:
            score += 3  # strong exact match in title
        if query_text.lower() in body:
            score += 2  # strong exact match in body  # boost for actual keyword match in body text
        if department in [d.lower() for d in selected_departments]:
            score += 3
        if has_important_link:
            score += 2

        return score

    # Score, filter, and sort documents
    scored_docs = [(doc, score(doc)) for doc in initial_results]
    filtered = [doc for doc, s in scored_docs if s >= 4]
    # Sort the filtered results based on their relevance scores
    ranked = sorted(filtered, key=lambda d: score(d), reverse=True)
    # Dynamically select top N chunks with strong scores
    high_confidence_docs = []
    for doc, s in scored_docs:
        if s >= 6:  # very strong
            high_confidence_docs.append(doc)
        elif s >= 4 and len(high_confidence_docs) < 3:
            high_confidence_docs.append(doc)
        if len(high_confidence_docs) >= 3:
            break

    return high_confidence_docs

# Generate Lee's response using streaming mode with caching
def generate_response_with_mistral(query, context, injected_link=None):

    # Build conversation history context from previous user/assistant exchanges
    messages = [
        {"role": "system", "content": (
            "You are Lee Knows, a helpful assistant working for Knowsley Council. Answer customer questions in a friendly and professional tone using UK English. Be clear and concise. Do not include greetings or sign-offs. Prioritise the most relevant and specific information first. Avoid repeating the question. If action is required, show the most relevant option first. Only use information and links that are explicitly provided in the context below. Do not make up URLs, contact details, or service descriptions. Do not invent documents, surveys, or links unless explicitly included."
        )}
    ]

    # Include up to 5 past exchanges in memory
    memory_limit = 5
    history = list(reversed(st.session_state.log))[:memory_limit * 2]  # includes both user and Lee turns
    history.reverse()

    for role, content in history:
        if role == "You":
            messages.append({"role": "user", "content": content})
        elif role == "Lee":
            messages.append({"role": "assistant", "content": content})

    # Append the new query with current context
    messages.append({"role": "user", "content": f"""User asked: {query}

Here is the relevant context:
{context}"""})

    stream = ollama.chat(model="mistral", messages=messages, stream=True)
    output = injected_link + "\n\n" if 'injected_link' in locals() else ""

    placeholder = st.empty()
    for chunk in stream:
        content = chunk['message']['content']
        output += content
        placeholder.markdown(output)
        sleep(0.01)
    placeholder.empty()
    return output

# User input and submit form (allows pressing Enter to submit)
frequently_asked = [
    "How do I report a missed bin collection?",
    "How can I apply for or renew my council tax discount or exemption?",
    "Where can I report potholes or damage to roads and pavements?",
    "How do I apply for housing or join the housing register?",
    "What do I need to do to get a permit for parking or visitor parking?",
    "How can I report anti-social behaviour or noise complaints?",
    "Where can I find information about school admissions or catchment areas?",
    "How do I apply for benefits like Housing Benefit or Council Tax Support?",
    "What’s the process for reporting fly-tipping or illegal dumping?",
    "How do I book a bulky waste collection or dispose of large household items?"
]

if "selected_faq" not in st.session_state:
    st.session_state.selected_faq = ""

with st.expander("Frequently Asked Questions", expanded=False):
    for faq in frequently_asked:
        if st.button(faq):
            st.session_state.chat_input = faq
            st.session_state.auto_submit = True
            st.session_state.scroll_to_answer = True
            st.rerun()

with st.form("query_form"):
    query = st.text_input("Ask Lee a question", key="chat_input", placeholder="Type your question and hit Enter")
    submit = st.form_submit_button("Ask")
auto_submit = st.session_state.pop("auto_submit", False)



# Process the query if submitted or triggered by FAQ
if (submit or auto_submit) and query.strip() and query != st.session_state.last_query:
    st.session_state.avatar_thinking = True
    with st.spinner("Lee is thinking..."):
        results = search_faiss(query)
        injected_link = ""  # Ensure it's always defined
        if results:
            context_parts = []
            for doc in results:
                context_parts.append(doc['text'])
                if doc.get("links"):
                    for link in doc["links"]:
                      if link.get("important"):
                        context_parts.append(f"- {link['text']}: {link['url']}")
            # Select best link and collect action links from used docs
            best_link = None
            best_score = 0
            action_links = []
            for doc in results:
                for link in doc.get("links", []):
                    if link.get("important"):
                        action_links.append((link["text"], link["url"]))
                        
            # Join all context parts into a single string with spacing for Mistral
            context = "\n\n".join(context_parts)

            # Inject best link if one was found
            if best_link:
                injected_link = f"You can {best_link['text'].lower()} [here]({best_link['url']})."
                context = f"""IMPORTANT: If referencing a link, use this one:
- {best_link['text']}: {best_link['url']}

""" + context
                        # Insert best link into context before generating response
            if best_link:
                context = f"""IMPORTANT: If referencing a link, use this one:
- {best_link['text']}: {best_link['url']}

""" + context

            answer = generate_response_with_mistral(query, context, injected_link)

            docs_used = results  # These are the actual chunks used to create the context
            sources = [
                f"<a href='{doc.get('url', '#')}' target='_blank'>{doc.get('title', 'View Source')}</a>"
                for doc in docs_used if doc.get('url')
            ]

            styled_sources = """
<details style='margin-top:10px; padding:10px; background-color:#f9f9f9; border-left: 4px solid #4a90e2;'>
<summary><strong>Sources</strong></summary>
<ul style='padding-left: 20px;'>
"""
            for link in sources:
                styled_sources += f"<li>{link}</li>"
            styled_sources += "</ul></details>"

            # Collect important links from sources and pick the best match for injection
            best_link = None
            best_score = 0
            action_links = []
            for doc in results:
                for link in doc.get("links", []):
                    if link.get("important"):
                        action_links.append((link["text"], link["url"]))
                        link_text = link["text"].lower()
                        score = sum(1 for word in query.lower().split() if word in link_text)
                        if score > best_score:
                            best_score = score
                            best_link = link

                            context = f"""IMPORTANT: If referencing a link, use this one:
- {best_link['text']}: {best_link['url']}

""" + context
            for doc in docs_used:
                for link in doc.get("links", []):
                    if link.get("important"):
                        action_links.append((link["text"], link["url"]))

            
            # Combine Lee's response, link, and sources into the full reply block
            response_block = f"""{answer}"""
            if action_links:
                # Deduplicate action links
                seen = set()
                unique_links = [(text, url) for text, url in action_links if not (url in seen or seen.add(url))]
                response_block += """
<div style='margin-top:15px; padding:10px; background-color:#eef9f1; border-left: 4px solid #3bb273;'>
<strong>Take action:</strong>
<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>
"""
                for text, url in unique_links:
                    response_block += f"<a href='{url}' target='_blank' style='text-decoration: none;'><button style='background-color: #3bb273; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer;'>{text}</button></a>"
                response_block += "</div></div>"
            response_block += f"{styled_sources}"
            st.session_state.chat = [("Lee", response_block)]
            st.session_state.log.insert(0, ("Lee", response_block))
            st.session_state.log.insert(0, ("You", query))
        else:
            st.session_state.chat = [("Lee", "Sorry, I couldn’t find anything on that topic. Please try rewording your question."), ("You", query)]
            st.session_state.log.insert(0, ("Lee", "Sorry, I couldn’t find anything on that topic. Please try rewording your question."))
            st.session_state.log.insert(0, ("You", query))
    st.session_state.avatar_thinking = False

# Scroll to latest response
    st.markdown("<div id='lee-response'></div>", unsafe_allow_html=True)
    if st.session_state.get("scroll_to_answer"):
        st.markdown("""
        <script>
        const leeResponse = document.getElementById('lee-response');
        if (leeResponse) {
            leeResponse.scrollIntoView({ behavior: 'smooth' });
        }
        </script>
        """, unsafe_allow_html=True)
        st.session_state.scroll_to_answer = False

# Display the latest response
if st.session_state.chat:
    st.markdown("---")
    st.subheader(f"You asked: {query}")
    st.markdown("""
    <script>
    const leeResponse = document.getElementById('lee-response');
    if (leeResponse) {
        leeResponse.scrollIntoView({ behavior: 'smooth' });
    }
    </script>
    """, unsafe_allow_html=True)
    for sender, message in st.session_state.chat:
        st.markdown(f"**{sender}:**", unsafe_allow_html=True)
        st.markdown(message, unsafe_allow_html=True)

# Chat history controls
st.markdown("---")
st.subheader("Conversation Controls")
col1, col2 = st.columns(2)
with col1:
    show_history = st.checkbox("Show Full Conversation History")
with col2:
    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.session_state.log = []

# Show full chat history if enabled
if show_history:
    st.markdown("---")
    st.subheader("Conversation History")
    for sender, message in reversed(st.session_state.log):
        st.markdown(f"**{sender}:**", unsafe_allow_html=True)
        st.markdown(message, unsafe_allow_html=True)
