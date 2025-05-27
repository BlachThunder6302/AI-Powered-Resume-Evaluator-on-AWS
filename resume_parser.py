import boto3
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import datetime
import hashlib
import os
from decouple import config

# Version
__version__ = "20250430_v23"  # Updated version

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def clean_text(text):
    # Minimal cleaning: remove emails and URLs only
    text = re.sub(r'^[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}$|^[A-Z]{2,}(?: [A-Z]{2,}){1,3}$', '', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    return text.strip()

def chunk_text(text):
    # Robust regex for section headers
    headers = r'(?i)(?:\n\n|\n|^)(Skills|Technical Skills|Key Skills|Proficiencies|Experience|Work Experience|Professional Experience|Education|Academic Background|Projects|Project Experience)[\s:]*\n'
    sections = re.split(headers, text)
    result = []
    for i in range(1, len(sections), 2):
        result.append(f"{sections[i]}\n{sections[i+1].strip()}")
    if not result:
        # Fallback: Split into ~500-char chunks
        result = [text[i:i+500] for i in range(0, len(text), 500)]
    return [s.strip() for s in result if s.strip()]

def upload_to_s3(file_path, bucket, object_name, aws_access_key, aws_secret_key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name='us-east-1'
    )
    try:
        s3.upload_file(file_path, bucket, object_name)
        url = f"https://{bucket}.s3.amazonaws.com/{object_name}"
        return url
    except Exception as e:
        raise Exception(f"S3 upload failed: {str(e)}")
    
def extract_text_from_pdf(pdf_path, aws_access_key, aws_secret_key):
    textract = boto3.client(
        'textract',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name='us-east-1'
    )
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name='us-east-1'
    )
    bucket = 'sre-bucket-24165409'
    s3_key = f"resumes/resume_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    s3.upload_file(pdf_path, bucket, s3_key)
    response = textract.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': s3_key}}
    )
    job_id = response['JobId']
    while True:
        result = textract.get_document_text_detection(JobId=job_id)
        if result['JobStatus'] == 'SUCCEEDED':
            break
        elif result['JobStatus'] == 'FAILED':
            raise Exception("Textract job failed")
    text = ""
    for item in result['Blocks']:
        if item['BlockType'] == 'LINE':
            text += item['Text'] + "\n"
    return text

def extract_details(resume_text, debug_info=None):
    if debug_info is None:
        debug_info = {}
    details = {'Name': '', 'Email': '', 'Projects': []}
    lines = resume_text.split('\n')
    name_pattern = r'\b(?:[A-Z]\.? ?){1,4}[A-Z][a-zA-Z]{2,}(?: [A-Z][a-zA-Z]{2,})?\b'  # Updated regex
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    project_pattern = r'^(.*?): Developed.*?|^(.*?): Conducted.*?|^(.*?): A web.*?'
    
    for i, line in enumerate(lines):
        if not details['Name']:
            match = re.search(name_pattern, line.strip())
            if match:
                details['Name'] = match.group(0)
                debug_info['extracted_name'] = f"Extracted Name: {details['Name']}"
        if not details['Email'] and re.search(email_pattern, line):
            details['Email'] = re.search(email_pattern, line).group()
        if re.match(project_pattern, line):
            project_name = re.match(project_pattern, line).group(1)
            if project_name:
                details['Projects'].append(project_name.strip())
    
    return details

def get_embedding(text, debug_info=None):
    bedrock = boto3.client(
        'bedrock-runtime',
        aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/embed_{text_hash}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            debug_info['embed_cache'] = f"Loaded embedding from cache: {cache_file}"
            return np.array(json.load(f))
    
    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps({
                "inputText": text,
                "dimensions": 1024,
                "normalize": True
            })
        )
        result = json.loads(response['body'].read().decode())
        embedding = np.array(result['embedding'])
        with open(cache_file, 'w') as f:
            json.dump(embedding.tolist(), f)
        debug_info['embed_saved'] = f"Saved embedding to cache: {cache_file}"
        return embedding
    except Exception as e:
        debug_info['embed_error'] = f"Error generating embedding: {str(e)}"
        return None

def calculate_ats_score(resume_text, jd_text):
    debug_info = {
        'version': __version__,
        'resume_clean': clean_text(resume_text)[:200],
        'jd_clean': clean_text(jd_text)[:200],
        'resume_sections': len(chunk_text(resume_text)),
        'jd_sections': len(chunk_text(jd_text))
    }
    
    # Minimal cleaning
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    
    # Chunk text
    resume_sections = chunk_text(resume_text)
    jd_sections = chunk_text(jd_text)
    
    # Compute embeddings for sections
    resume_embeddings = [get_embedding(clean_text(section), debug_info) for section in resume_sections]
    jd_embeddings = [get_embedding(clean_text(section), debug_info) for section in jd_sections]
    
    # Filter out None embeddings
    resume_embeddings = [e for e in resume_embeddings if e is not None]
    jd_embeddings = [e for e in jd_embeddings if e is not None]
    
    if not resume_embeddings or not jd_embeddings:
        debug_info['embedding_error'] = "Failed to generate embeddings"
        return {
            'score': 0.0,
            'resume_tokens': [],
            'jd_tokens': [],
            'tfidf_matrix_shape': (0, 0),
            'cosine_sim': 0.0,
            'feature_names': [],
            'debug_info': debug_info
        }
    
    # Compute max cosine similarity
    max_cosine = 0.0
    for resume_emb in resume_embeddings:
        for jd_emb in jd_embeddings:
            cosine = cosine_similarity([resume_emb], [jd_emb])[0][0]
            max_cosine = max(max_cosine, cosine)
    
    ats_score = max_cosine * 150
    ats_score = min(ats_score, 70.0)
    
    resume_tokens = resume_clean.lower().split()[:20]
    jd_tokens = jd_clean.lower().split()[:20]
    feature_names = []
    debug_info['resume_tokens'] = resume_tokens
    debug_info['jd_tokens'] = jd_tokens
    debug_info['cosine_sim'] = f"{max_cosine:.3f}"
    
    return {
        'score': ats_score,
        'resume_tokens': resume_tokens,
        'jd_tokens': jd_tokens,
        'tfidf_matrix_shape': (2, 1024),
        'cosine_sim': max_cosine,
        'feature_names': feature_names,
        'debug_info': debug_info
    }