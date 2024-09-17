import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import re
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import requests
import multiprocessing
import logging
from configparser import ConfigParser
import os

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Load SpaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load configuration
config = ConfigParser()
config.read('config.ini')

# Set up logging
logging.basicConfig(filename='analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_text(text):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def extract_drug_info(text):
    doc = nlp(text)
    drug_entities = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]
    
    # Use regex to find potential drug names not caught by SpaCy
    regex_drugs = re.findall(r'\b[A-Z][a-z]+(?:-[A-Z][a-z]+)*(?:-\d+)?\b', text)
    
    return list(set(drug_entities + regex_drugs))

def determine_trial_phase(text):
    phase_patterns = {
        'Preclinical': r'\bpreclinical\b',
        'Phase 1': r'\bphase 1\b|\bphase i\b',
        'Phase 2': r'\bphase 2\b|\bphase ii\b',
        'Phase 3': r'\bphase 3\b|\bphase iii\b',
        'FDA Approved': r'\bfda approved\b'
    }
    
    for phase, pattern in phase_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return phase
    return 'Unknown'

def get_clinical_trial_data(drug_name):
    url = f"https://clinicaltrials.gov/api/query/study_fields?expr={drug_name}&fields=Phase,BriefTitle,OverallStatus&fmt=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.warning(f"Failed to fetch data for {drug_name}: {str(e)}")
        return None

def perform_topic_modeling(texts, n_topics=5):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(doc_term_matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    return [
        [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        for topic in lda.components_
    ]

def analyze_10k_filing(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        preprocessed_content = preprocess_text(content)
        sentences = sent_tokenize(content)
        
        drug_names = extract_drug_info(content)
        
        drugs = []
        for drug in set(drug_names):
            drug_sentences = [s for s in sentences if drug.lower() in s.lower()]
            if drug_sentences:
                phase = determine_trial_phase(' '.join(drug_sentences))
                
                # Get additional data from ClinicalTrials.gov
                clinical_data = get_clinical_trial_data(drug)
                
                drugs.append({
                    'name': drug,
                    'description': ' '.join(drug_sentences),
                    'phase': phase,
                    'clinical_data': clinical_data
                })
        
        # Perform topic modeling on the entire document
        topics = perform_topic_modeling([preprocessed_content])
        
        return {
            'file_path': file_path,
            'drugs': drugs,
            'topics': topics
        }
    
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def create_database():
    conn = psycopg2.connect(
        dbname=config.get('Database', 'dbname'),
        user=config.get('Database', 'user'),
        password=config.get('Database', 'password'),
        host=config.get('Database', 'host')
    )
    cur = conn.cursor()
    
    # Create tables
    cur.execute('''
        CREATE TABLE IF NOT EXISTS filings (
            id SERIAL PRIMARY KEY,
            file_path TEXT UNIQUE,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS drugs (
            id SERIAL PRIMARY KEY,
            filing_id INTEGER REFERENCES filings(id),
            name TEXT,
            description TEXT,
            phase TEXT
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id SERIAL PRIMARY KEY,
            filing_id INTEGER REFERENCES filings(id),
            topic_words TEXT[]
        )
    ''')
    
    conn.commit()
    return conn

def insert_analysis_results(conn, results):
    cur = conn.cursor()
    
    cur.execute("INSERT INTO filings (file_path) VALUES (%s) RETURNING id", (results['file_path'],))
    filing_id = cur.fetchone()[0]
    
    for drug in results['drugs']:
        cur.execute('''
            INSERT INTO drugs (filing_id, name, description, phase)
            VALUES (%s, %s, %s, %s)
        ''', (filing_id, drug['name'], drug['description'], drug['phase']))
    
    for topic in results['topics']:
        cur.execute('''
            INSERT INTO topics (filing_id, topic_words)
            VALUES (%s, %s)
        ''', (filing_id, topic))
    
    conn.commit()

def main():
    filing_paths = [os.path.join(config.get('Filings', 'directory'), f) for f in os.listdir(config.get('Filings', 'directory')) if f.endswith('.txt')]
    
    conn = create_database()
    
    with multiprocessing.Pool() as pool:
        results = pool.map(analyze_10k_filing, filing_paths)
    
    for result in results:
        if result:
            insert_analysis_results(conn, result)
    
    conn.close()
    logging.info("Analysis complete. Results stored in the database.")

if __name__ == "__main__":
    main()
