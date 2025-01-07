import imaplib
import email
from email.header import decode_header
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
from datetime import datetime
from database import create_connection, insert_email, email_exists, update_email_processed
from nltk.corpus import stopwords
import nltk
from dateutil import parser
import pandas as pd
from langdetect import detect
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# Load the small English NLP model
nlp = spacy.load("en_core_web_sm")

# Download stopwords if you haven't already
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Convert stop words to a list
stop_words_list = list(stop_words)

def load_training_data(file_path):
    """Load training data from a CSV file."""
    print(f"Loading training data from {file_path}")
    try:
      df = pd.read_csv(file_path, encoding='utf-8') # Specify encoding here
      print(f"File loaded correctly, number of rows: {len(df)}")
      print(f"Text sample: {df['text'].iloc[0]}")
      texts = df['text'].tolist()
      languages = df['language'].tolist()
      for i, text in enumerate(texts):
          if not isinstance(text, str):
              print(f"Error: Text at index {i} is not a string: {text}, type: {type(text)}")
          texts[i] = clean_text(text)
      return texts, df['category'].tolist(), df['priority'].tolist(), languages
    except Exception as e:
        print(f"Error loading the training dataset {e}")
        return [],[],[],[]

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def detect_language(text):
     try:
        language = detect(text)
        return language
     except:
        return "unknown"

# Load training data from CSV
training_file = 'email_training_data_multilingual.csv'
texts, categories, priorities, languages = load_training_data(training_file)

if not texts:
    print("Error: training data is empty.")
else:
    # Create vectorizer for texts
    vectorizer = TfidfVectorizer(stop_words=stop_words_list)
    features = vectorizer.fit_transform(texts)

    # Train models
    category_model = LogisticRegression()
    category_model.fit(features, categories)

    priority_model = LogisticRegression()
    priority_model.fit(features, priorities)


def get_email_data():
    """Fetches emails and extracts required data"""

    email_address = os.getenv("EMAIL_ADDRESS")
    email_password = os.getenv("EMAIL_PASSWORD")
    imap_server = os.getenv("IMAP_SERVER")
    conn = create_connection()

    if not all([email_address, email_password, imap_server]):
         print("Error: Email credentials missing.")
         return

    print("Attempting to connect to the IMAP server...") #Log before connecting

    try:
         # Connect to the IMAP server
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_address, email_password)
        mail.select("inbox")

        print("Successfully connected to the IMAP server.") #Log after connecting

         # Search for unseen emails
        _, data = mail.search(None, "UNSEEN")
        email_ids = data[0].split()

        for email_id in email_ids:
            if not email_id:
                continue
            #Check if email already exists in database
            if email_exists(conn, email_id.decode()):
                update_email_processed(conn, email_id.decode())
                print(f"Email with id {email_id.decode()} already processed and marked as so.")
                continue

            _, msg_data = mail.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                 if isinstance(response_part, tuple):
                      msg = email.message_from_bytes(response_part[1])
                      email_body = ""

                      if msg.is_multipart():
                            for part in msg.walk():
                                content_type = part.get_content_type()
                                content_disposition = str(part.get("Content-Disposition"))

                                if "attachment" not in content_disposition:
                                    try:
                                        if content_type == "text/plain":
                                          email_body += part.get_payload(decode=True).decode()
                                        elif content_type == "text/html":
                                          email_body += part.get_payload(decode=True).decode()
                                    except:
                                          pass

                      else:
                          email_body = msg.get_payload(decode=True).decode()


                      email_body = email_body.replace('\r\n', ' ').replace('\n',' ')

                      email_date = msg.get("Date")
                      if email_date:
                         email_date = parser.parse(email_date).isoformat()
                      else:
                          email_date = datetime.now().isoformat()

                      language = detect_language(email_body)
                      print(f"Detected language: {language}")


                      contact_name = extract_name(email_body)
                      company = extract_company(email_body)
                      email_address = extract_email(email_body)
                      phone_number = extract_phone(email_body)

                      email_text_features = vectorizer.transform([email_body])
                      category = category_model.predict(email_text_features)[0]
                      priority = priority_model.predict(email_text_features)[0]

                      email_data = (email_id.decode(), contact_name, company, email_address, phone_number, category, priority, email_body, email_date, 0, language)

                      insert_email(conn, email_data)

                      print(f"Email from: {contact_name} processed and inserted")

                      #Send summary Email
                      summary = f"Contact Name: {contact_name}\nCompany: {company}\nEmail Address: {email_address}\nPhone Number: {phone_number}\nCategory: {category}\nPriority: {priority}\nLanguage:{language}\nDate:{email_date}"
                      send_summary_email("mgrisoster@gmail.com", summary, email_address, email_password)
                      #mark as read
                      #mail.store(email_id, "+FLAGS", "\\Seen")
    except Exception as e:
             print(f"An error occurred: {e}")
    finally:
            if conn:
                conn.close()
            if 'mail' in locals() and mail:
                mail.logout()

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
      if ent.label_ == "PERSON":
         return ent.text
    return "Not found"

def extract_company(text):
    # Simple keyword-based company name detection
    keywords = ['company', 'organization', 'inc', 'ltd', 'corp']
    for keyword in keywords:
      match = re.search(rf"(\w+)\s+{keyword}",text, re.IGNORECASE)
      if match:
         return match.group(1).capitalize()
    return "Not found"

def extract_email(text):
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return email_match.group(0) if email_match else "Not found"


def extract_phone(text):
    phone_match = re.search(r'[\d\-\.\s()+]{7,}',text)
    return phone_match.group(0) if phone_match else "Not found"

def send_summary_email(recipient_email, summary, email_address, email_password):
    """Sends a summary email with the identified features."""

    sender_email = email_address
    sender_password = email_password

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = 'Email Processing Summary'
    message.attach(MIMEText(summary, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(message)
        print(f"Summary email sent to {recipient_email}")
    except Exception as e:
        print(f"Error sending email: {e}")
from database import get_emails, create_connection

if __name__ == '__main__':
   get_email_data()
   conn = create_connection()
   if conn:
     emails = get_emails(conn, False)
     df = pd.DataFrame(emails, columns = ["id","email_id","contact_name","company","email_address","phone_number","category","priority","email_body","email_date","processed", "language"])
     print(df)
     conn.close()