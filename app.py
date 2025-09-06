import os
import re
import base64
import json
import time
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from textblob import TextBlob
from email.mime.text import MIMEText
from dotenv import load_dotenv
from collections import Counter

# --- LangChain Imports ---
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Environment Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Flask App Initialization ---
app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here'

# --- CONFIGURATION ---
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send'
]
SUPPORT_KEYWORDS = ['support', 'query', 'request', 'help']
URGENT_KEYWORDS = ['urgent', 'critical', 'immediately', 'down', 'failed', 'cannot access']
MAX_EMAIL_CHARS = 800   # reduced to avoid token overload

# --- Global variable for the RAG chain and LLM ---
rag_chain = None
llm = None


# ========== Retry Wrapper ==========
def safe_invoke(chain, input_data, retries=3, wait=10):
    for i in range(retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"[WARN] Rate limit hit. Retrying in {wait}s... ({i+1}/{retries})")
                time.sleep(wait)
            else:
                raise
    return "Error: Could not generate reply after retries."


# ========== Setup RAG ==========
def setup_rag_pipeline():
    """Sets up the LangChain RAG pipeline and a general LLM instance."""
    global rag_chain, llm
    print("Setting up RAG pipeline...")

    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
        with open("knowledge_base/example.txt", "w") as f:
            f.write("This is a placeholder knowledge base. Add .txt files here for better replies.")

    loader = DirectoryLoader('knowledge_base/', glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = FAISS.from_documents(texts, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    prompt_template = """
    You are a professional and friendly customer support assistant. 
    Use the following CONTEXT from our knowledge base to draft a helpful and empathetic reply. 

    CONTEXT: {context}
    CUSTOMER EMAIL: Subject: {subject}, From: {sender}, Body: {question}
    INSTRUCTIONS: {empathy_instruction} 
    Answer the customer's question using the context. 
    If the answer is not in the context, say you will escalate the issue. 
    Sign off as "The Support Team".

    DRAFT REPLY:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "subject", "sender", "question", "empathy_instruction"]
    )

    # ✅ Updated model
    llm = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough(), "subject": RunnablePassthrough(),
         "sender": RunnablePassthrough(), "empathy_instruction": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG pipeline setup complete.")


# ========== Extraction ==========
def extract_key_information(email_body):
    """Extracts key information from email body using an LLM."""
    global llm
    if not llm:
        return {"error": "LLM not initialized."}

    extraction_prompt_template = """
    You are an expert at extracting structured information from unstructured text.
    Analyze the following customer email body and extract the information in a JSON format:
    1. "customer_name": The name of the customer, if mentioned.
    2. "contact_phone": Any phone numbers mentioned.
    3. "contact_email": Any alternate email addresses mentioned.
    4. "summary": A brief, one-sentence summary of the customer's main request or problem.
    5. "sentiment_keywords": A list of specific words or short phrases that indicate the customer's sentiment.

    EMAIL BODY: {email_body}

    Provide only the JSON object as your response. If a piece of information is not present, use a null or empty value.
    """
    prompt = PromptTemplate(template=extraction_prompt_template, input_variables=["email_body"])
    extraction_chain = prompt | llm | StrOutputParser()

    try:
        response_str = safe_invoke(extraction_chain, {"email_body": email_body})
        return json.loads(response_str.strip().replace("```json", "").replace("```", ""))
    except Exception as e:
        print(f"Error during LLM extraction: {e}")
        return {
            "customer_name": "N/A", "contact_phone": [], "contact_email": [],
            "summary": "Could not generate summary.", "sentiment_keywords": []
        }


# ========== Draft Reply ==========
def generate_draft_reply(email_data):
    """Generates a draft reply using the RAG chain."""
    global rag_chain
    if not rag_chain:
        return "Error: RAG pipeline not initialized."

    empathy_instruction = ""
    if email_data.get('sentiment') == 'Negative':
        empathy_instruction = "The customer seems frustrated. Start by acknowledging their frustration empathetically."

    input_data = {
        "question": email_data['body'][:MAX_EMAIL_CHARS],
        "subject": email_data['subject'],
        "sender": email_data['sender'],
        "empathy_instruction": empathy_instruction
    }

    try:
        return safe_invoke(rag_chain, input_data)
    except Exception as e:
        return f"Error generating reply: {str(e)}"


# --- Gmail Helpers ---
def connect_to_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)


def get_email_body(payload):
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data:
                    return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            elif 'parts' in part:
                body = get_email_body(part)
                if body:
                    return body
    elif payload['mimeType'] == 'text/plain':
        data = payload['body'].get('data')
        if data:
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
    return ""


def analyze_and_rank_email(email_data):
    score = 0
    full_text_lower = (email_data['subject'].lower() + " " + email_data.get('snippet', '').lower())

    for keyword in URGENT_KEYWORDS:
        if keyword in full_text_lower:
            score += 10
            break

    blob = TextBlob(email_data.get('snippet', ''))
    polarity = blob.sentiment.polarity
    if polarity < -0.1:
        score += 3
        email_data['sentiment'] = 'Negative'
    elif polarity > 0.1:
        email_data['sentiment'] = 'Positive'
    else:
        email_data['sentiment'] = 'Neutral'

    email_data['priority_score'] = score
    email_data['priority'] = 'Urgent' if score >= 10 else 'Normal'

    try:
        date_str = email_data.get('date', '')
        cleaned_date_str = re.sub(r'\s*\([^)]*\)', '', date_str).strip()
        dt_object = datetime.strptime(cleaned_date_str, '%a, %d %b %Y %H:%M:%S %z')
        email_data['datetime'] = dt_object
    except (ValueError, TypeError):
        email_data['datetime'] = None
    return email_data


# --- Flask Routes ---
@app.route('/')
def dashboard():
    service = connect_to_gmail()
    query = " OR ".join([f"subject:({keyword})" for keyword in SUPPORT_KEYWORDS])
    results = service.users().messages().list(
        userId='me', labelIds=['INBOX'], q=query, maxResults=20
    ).execute()
    messages = results.get('messages', [])

    if not messages:
        return render_template('index.html', emails=[], stats=json.dumps({}))

    email_list = []
    for msg_info in messages:
        msg = service.users().messages().get(
            userId='me', id=msg_info['id'], format='metadata',
            metadataHeaders=['Subject', 'From', 'Date']
        ).execute()
        headers = msg.get('payload', {}).get('headers', [])
        email_data = {
            'id': msg['id'], 'snippet': msg.get('snippet', ''),
            'subject': next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject'),
            'sender': next((h['value'] for h in headers if h['name'].lower() == 'from'), ''),
            'date': next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
        }

        subject_lower = email_data['subject'].lower()
        if any(keyword in subject_lower for keyword in SUPPORT_KEYWORDS):
            email_list.append(analyze_and_rank_email(email_data))

    sorted_emails = sorted(email_list, key=lambda e: e['priority_score'], reverse=True)

    now = datetime.now(timezone.utc)
    last_24_hours = now - timedelta(days=1)
    emails_last_24h = [e for e in email_list if e['datetime'] and e['datetime'] > last_24_hours]

    stats = {
        'total_emails': len(email_list), 'total_last_24h': len(emails_last_24h),
        'pending': len(email_list), 'resolved': 0,
        'priority_counts': Counter(e['priority'] for e in email_list),
        'sentiment_counts': Counter(e['sentiment'] for e in email_list)
    }

    return render_template('index.html', emails=sorted_emails, stats=json.dumps(stats))


@app.route('/get_email_details/<message_id>')
def get_email_details(message_id):
    service = connect_to_gmail()
    msg = service.users().messages().get(userId='me', id=message_id).execute()
    payload = msg.get('payload', {})
    headers = payload.get('headers', [])

    body = get_email_body(payload)
    email_data = {
        'id': msg['id'], 'threadId': msg['threadId'],
        'subject': next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject'),
        'sender': next((h['value'] for h in headers if h['name'].lower() == 'from'), ''),
        'body': body
    }

    email_data['sentiment'] = 'Negative' if TextBlob(body).sentiment.polarity < -0.1 else 'Neutral'
    email_data['draft_reply'] = generate_draft_reply(email_data)
    email_data['extracted_info'] = extract_key_information(body)

    return jsonify(email_data)


@app.route('/send_reply', methods=['POST'])
def send_reply():
    service = connect_to_gmail()
    form_data = request.form

    message = MIMEText(form_data['body'])
    message['to'] = form_data['recipient']
    message['subject'] = form_data['subject']

    create_message = {
        'raw': base64.urlsafe_b64encode(message.as_bytes()).decode(),
        'threadId': form_data['threadId']
    }

    try:
        service.users().messages().send(userId='me', body=create_message).execute()
        flash(f"Success! Reply sent to {form_data['recipient']}", "success")
    except Exception as e:
        flash(f"Error sending email: {e}", "error")
    return redirect(url_for('dashboard'))


# ========== Flask Entry ==========
if __name__ == '__main__':
    setup_rag_pipeline()
    app.run(debug=True)
