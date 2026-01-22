# Realtime Chat Bot with Knowledge Base

This is an enhanced version of the chat bot that uses OpenAI's Realtime API for voice interaction, combined with the existing knowledge base functionality including embeddings, fact extraction, and semantic search.

## Features

- **Voice Interaction**: Real-time voice input and output using OpenAI's Realtime API
- **Knowledge Base**: Stores facts extracted from conversations with embeddings
- **Semantic Search**: Finds relevant information from past conversations
- **Fact Extraction**: Automatically extracts structured facts (people, objects, importance) from user statements
- **Persistent Storage**: Saves conversations and fact summaries to JSON files
- **Vector Search**: Uses embeddings for fast similarity search (with sqlite-vec when available)

## Requirements

- Python 3.8+
- OpenAI API key with Realtime API access
- Microphone and speakers for voice interaction

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the realtime chat bot:
```bash
python realtime_chat.py
```

### Controls
- Speak naturally - the bot will listen and respond with voice
- Press 'q' to quit and save the conversation

### Tools Available
- **search_facts**: Search for similar facts in the knowledge base
- **save_conversation**: Manually save the current conversation and extract facts

## File Structure

- `realtime_chat.py`: Main realtime chat application
- `chat.py`: Original text-based chat bot
- `dialogues/`: Saved conversation files
- `summaries/`: Extracted facts and summaries
- `embeddings_db/`: SQLite database with fact embeddings

## How It Works

1. **Voice Input**: Uses OpenAI Realtime API for continuous voice streaming
2. **Conversation Tracking**: Maintains conversation history in memory
3. **Fact Extraction**: Periodically extracts structured facts from user statements
4. **Embedding Generation**: Creates embeddings for facts using OpenAI's text-embedding-3-small
5. **Semantic Search**: Uses cosine similarity to find relevant past information
6. **Tool Integration**: Provides search and save functionality through Realtime API tools

## Data Storage

- **Dialogues**: Full conversation transcripts saved as JSON
- **Facts**: Structured facts with people, objects, and importance scores
- **Embeddings**: Vector representations for semantic search
- **Summaries**: Extracted conversation summaries

## Comparison with Original Chat Bot

| Feature | Original (chat.py) | Realtime (realtime_chat.py) |
|---------|-------------------|----------------------------|
| Input Method | Text (speech recognition) | Voice (Realtime API) |
| Output Method | Text | Voice |
| Real-time | No | Yes |
| Tools | None | Search & Save |
| Interface | Command line | Voice streaming |

The realtime version provides a more natural voice interface while maintaining all the knowledge base functionality of the original bot.