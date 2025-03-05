import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from embedding import initialize_embedder
from chat import ChatManager
import hmac

# Set page configuration - must be the first st command
st.set_page_config(
    page_title="Travel Advisor Chat",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

def check_password():
    """Returns `True` if the user had a correct password."""
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
        
    if st.session_state.password_correct:
        return True
    else:
        st.write("# Welcome to Travel Advisor Chat! 🌟")
        st.write("Please log in to continue.")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Log in"):
            if username in st.secrets["credentials"]:
                if hmac.compare_digest(password, st.secrets["credentials"][username]):
                    st.session_state.password_correct = True
                    st.rerun()  # Rerun the app to refresh the page after login
                    return True
            st.error("😕 User not recognized or password incorrect")
        return False

# Main app
if check_password():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please add it to your .env file as OPENAI_API_KEY=your_key_here")
        st.stop()

    # Initialize session state
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatManager()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "embedder" not in st.session_state:
        with st.spinner("Loading location data and initializing embeddings... This may take a minute on first run."):
            st.session_state.embedder = initialize_embedder("locations.csv")

    # Sidebar
    with st.sidebar:
        st.title("Travel Advisor Chat")
        st.markdown("Ask me about places to visit in Singapore! I can provide personalized recommendations based on your interests.")
        
        st.subheader("Example Questions")
        st.markdown("""
        - What are some good places to visit in Bedok?
        - I'm looking for cafes in Joo Chiat with a cozy atmosphere
        - Recommend a budget-friendly attraction for families
        - What's a good place for Vietnamese food?
        - Tell me about historical sites in Singapore
        """)
        
        st.subheader("About")
        st.markdown("""
        This application uses:
        - Local embedding of location data
        - Retrieval-Augmented Generation (RAG)
        - OpenAI's API for natural language responses
        
        The data includes information about attractions, cafes, restaurants, and more in Singapore.
        """)
        
        if st.button("Reset Conversation"):
            st.session_state.chat_manager.reset_conversation()
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    st.title("✈️ Singapore Travel Advisor")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about places to visit in Singapore..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Embed the query
                query_embedding = st.session_state.embedder.embed_query(prompt)
                
                # Retrieve similar locations
                similar_locations = st.session_state.embedder.retrieve_similar_locations(query_embedding, k=10)
                
                # Generate response
                response = st.session_state.chat_manager.generate_response(prompt, similar_locations)
                
                # Display response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Display information about the retrieved locations (hidden by default)
    if st.session_state.messages:  # Only show expander if there are messages
        with st.expander("View Retrieved Location Data", expanded=False):
            if "embedder" in st.session_state:
                last_user_message = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
                last_assistant_message = next((m["content"] for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)

                if last_user_message and last_assistant_message:
                    query_embedding = st.session_state.embedder.embed_query(last_user_message)
                    similar_locations = st.session_state.embedder.retrieve_similar_locations(query_embedding, k=10)
                    
                    # Filter locations that were mentioned in the response
                    mentioned_locations = []
                    for location in similar_locations:
                        # Create variations of the location name to check
                        location_name = location['title']
                        location_variations = [
                            location_name,
                            location_name.lower(),
                            location_name.replace("'", ""),  # Handle cases like "Maxwell's"
                            location_name.replace("'", "").lower(),
                            location_name.split(":")[0].strip(),
                            location_name.split(":")[0].strip().lower(),
                        ]
                        
                        # Check if any variation of the location name appears in the response
                        if any(variation in last_assistant_message or 
                              variation in last_assistant_message.lower() 
                              for variation in location_variations):
                            mentioned_locations.append(location)
                    
                    # Sort mentioned locations by similarity score
                    mentioned_locations.sort(key=lambda x: x['similarity_score'], reverse=True)
                    
                    if mentioned_locations:
                        for i, location in enumerate(mentioned_locations):
                            st.subheader(f"{i+1}. {location['title']}")
                            st.markdown(f"**Location:** {location['Location / Area']}")
                            st.markdown(f"**Category:** {location['Category / Type']}")
                            st.markdown(f"**Themes:** {location['Theme / Highlights']}")
                            st.markdown(f"**Similarity Score:** {location['similarity_score']:.4f}")
                            st.markdown("---")
                    else:
                        st.info("No specific locations were mentioned in the response.")