Intelligent Gmail Support Assistant
This project is a smart, automated system designed to streamline customer support by integrating directly with your Gmail inbox. It uses a powerful Large Language Model (LLM) with Retrieval-augmented Generation (RAG) to read, analyze, prioritize, and draft intelligent replies to support-related emails.

Key Features
Automated Email Fetching: Connects securely to your Gmail account to pull in emails marked with support-related keywords.

Intelligent Prioritization: Automatically analyzes emails for urgency and sentiment (using TextBlob and keyword matching) to rank them by priority.

RAG-Powered Draft Replies: Utilizes a LangChain RAG pipeline to generate context-aware draft replies. The model consults a local knowledge base (/knowledge_base) to provide accurate, helpful answers.

Information Extraction: The LLM extracts key information from emails, such as customer name, contact details, and a summary of the issue.

Web Dashboard: A clean, user-friendly dashboard built with Flask to view prioritized emails, review drafted replies, and send them directly from the interface.

Analytics Overview: The dashboard provides simple analytics, including total email volume, sentiment breakdown, and priority counts.

Technology Stack
This project is built with a modern stack of AI and web development tools:

Backend Framework: Flask

LLM Orchestration: LangChain

Large Language Model (LLM): Llama 3.3 70B via the Groq API

Embeddings: Ollama (all-minilm model)

Vector Store: FAISS (for in-memory similarity search)

Google Integration: Google API Python Client for Gmail

Sentiment Analysis: TextBlob

Environment Management: python-dotenv

Setup and Installation
Prerequisites
Python 3.8+

Google Cloud Platform project with the Gmail API enabled.

Ollama installed and running locally with the all-minilm model pulled (ollama pull all-minilm).

A Groq API Key.

1. Clone the Repository
git clone [https://github.com/your-username/intelligent-gmail-support.git](https://github.com/your-username/intelligent-gmail-support.git)
cd intelligent-gmail-support

2. Set up Google Authentication
Go to your Google Cloud Console.

Create a new project (or use an existing one).

Enable the Gmail API.

Create credentials for an OAuth 2.0 Client ID (select "Desktop app").

Download the credentials JSON file and save it as credentials.json in the project's root directory.

3. Install Dependencies
pip install -r requirements.txt

(Note: You will need to create a requirements.txt file based on the imports in your Python script.)

4. Configure Environment Variables
Create a .env file in the root directory and add your Groq API key:

GROQ_API_KEY="your-groq-api-key-here"

5. Create a Knowledge Base
Create a directory named knowledge_base in the root of the project. Add .txt files containing information you want the RAG system to use when drafting replies.

/knowledge_base
|-- product_info.txt
|-- faq.txt
|-- troubleshooting_steps.txt

6. Run the Application
The first time you run the app, you will be prompted to authorize access to your Gmail account through a browser window. This will create a token.json file for future sessions.

python your_main_script.py

Navigate to http://127.0.0.1:5000 in your web browser to access the dashboard.

How to Use
Dashboard View: Open the application to see a list of fetched support emails, sorted by priority.

View Details: Click on an email to see its full content, extracted information (like customer name and issue summary), and the AI-generated draft reply.

Edit and Send: Review the draft reply. You can edit it directly in the text area before clicking "Send Reply" to dispatch it from your connected Gmail account.

This project aims to be a powerful co-pilot for customer support teams, reducing response times and improving the quality of support interactions.