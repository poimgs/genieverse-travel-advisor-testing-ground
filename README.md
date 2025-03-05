# Travel Advisor Chat App

A chat application that provides personalized travel recommendations using Retrieval-Augmented Generation (RAG).

## Features

- Natural language chat interface for travel queries
- Local data embedding and retrieval from locations.csv
- Integration with OpenAI for generating responses
- Context-aware conversation that remembers user preferences
- Intuitive and user-friendly UI

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

1. Run the application:
```
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Start chatting with the travel advisor! You can ask questions like:
   - "What are some good places to visit in Bedok?"
   - "I'm looking for cafes in Joo Chiat with a cozy atmosphere"
   - "Recommend a budget-friendly attraction for families"
   - "What's a good place for Vietnamese food?"

## How It Works

1. **Data Embedding**: The application embeds location data from `locations.csv` using Sentence Transformers.
2. **Query Processing**: When you ask a question, it's embedded and compared to the location data.
3. **Retrieval**: The most relevant locations are retrieved based on vector similarity.
4. **Generation**: The retrieved context and your query are sent to OpenAI to generate a helpful response.
5. **Conversation Memory**: The app maintains context across multiple turns of conversation.

## Project Structure

- `app.py`: Main Streamlit application
- `embedding.py`: Functions for embedding and retrieving location data
- `chat.py`: Chat logic and OpenAI integration
- `locations.csv`: Dataset containing location information
- `requirements.txt`: Required Python packages
- `.env`: Environment variables (not tracked in git)

## Requirements

- Python 3.10.11
- OpenAI API key
