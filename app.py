import json
import streamlit as st
from resume_parser import extract_text_from_pdf, extract_details, calculate_ats_score, upload_to_s3
from decouple import config
import boto3
import re
import io
import datetime
import uuid
import hashlib
import os
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Cache directory
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_jd_with_bedrock(job_role, industry="General", debug_logs=None):
    bedrock = boto3.client(
        'bedrock-runtime',
        aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1',
        config=boto3.session.Config(
            retries={'max_attempts': 6, 'mode': 'adaptive'}
        )
    )
    role_hash = hashlib.md5(job_role.lower().encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/jd_{role_hash}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            debug_logs['jd_cache'] = f"Loaded JD from cache: {cache_file}"
            return json.load(f)
    
    prompt = f"""
    Generate a detailed job description for a {job_role} role in the {industry} industry. The output should be a JSON object with:
    {{
        "job_title": "<Job Title>",
        "responsibilities": ["<Responsibility 1>", ...],
        "required_skills": ["<Skill 1>", ...],
        "qualifications": ["<Qualification 1>", ...]
    }}
    Ensure the description is 400–500 words and the output is valid JSON. Return only the JSON object, without any introductory text.
    """
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.5,
                "top_p": 0.9,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        raw_response = response['body'].read().decode()
        debug_logs['bedrock_response'] = f"Raw Bedrock response: {raw_response[:200]}..."
        
        result = json.loads(raw_response)['content'][0]['text']
        debug_logs['jd_extracted'] = f"Extracted JD text: {result[:200]}..."
        
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = result
        
        try:
            raw_jd_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            debug_logs['jd_json_error'] = f"JSON parsing error: {str(e)}"
            json_str = json_str.strip()
            if not json_str.startswith('{'):
                json_str = '{' + json_str
            if not json_str.endswith('}'):
                json_str = json_str + '}'
            raw_jd_data = json.loads(json_str)
        
        jd_data = {
            "job_description": (
                f"Job Title: {raw_jd_data['job_title']}\n\n"
                f"Responsibilities:\n" + "\n".join(f"- {r}" for r in raw_jd_data['responsibilities']) + "\n\n"
                f"Qualifications:\n" + "\n".join(f"- {q}" for q in raw_jd_data['qualifications'])
            ),
            "skills": raw_jd_data['required_skills']
        }
        with open(cache_file, 'w') as f:
            json.dump(jd_data, f)
        debug_logs['jd_saved'] = f"Saved JD to cache: {cache_file}"
        return jd_data
    except Exception as e:
        debug_logs['bedrock_error'] = f"Bedrock JD generation failed: {str(e)}"
        st.error(f"❌ Bedrock JD generation failed: {str(e)}")
        return {"job_description": job_role, "skills": []}

def send_to_sqs(jd_text, cv_text, queue_url, debug_logs=None):
    sqs = boto3.client(
        'sqs',
        aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    message = {
        "jd_text": jd_text,
        "cv_text": cv_text,
        "timestamp": datetime.datetime.now().isoformat()
    }
    try:
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message),
            MessageGroupId="FeedbackQueue",
            MessageDeduplicationId=hashlib.md5((jd_text + cv_text).encode()).hexdigest()
        )
        debug_logs['sqs_message_id'] = f"SQS Message ID: {response['MessageId']}"
        return response['MessageId']
    except Exception as e:
        debug_logs['sqs_send_error'] = f"Failed to send to SQS: {str(e)}"
        st.error(f"❌ Failed to send to SQS: {str(e)}")
        return None

def get_titan_feedback(jd_text, cv_text, model_id="amazon.titan-text-premier-v1:0", debug_logs=None):
    bedrock = boto3.client(
        'bedrock-runtime',
        aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1',
        config=boto3.session.Config(
            retries={'max_attempts': 6, 'mode': 'adaptive'}
        )
    )
    input_text = jd_text[:2000] + cv_text[:2000]  # Truncate to avoid input limits
    text_hash = hashlib.md5(input_text.encode()).hexdigest()
    cache_file = f"{CACHE_DIR}/feedback_{text_hash}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            debug_logs['titan_cache'] = "Loaded Titan feedback from cache"
            return json.load(f)
    
    prompt = f"""
    Here is a job description:
    {jd_text[:2000]}
    
    Here is a candidate's resume:
    {cv_text[:2000]}
    
    On a scale of 0 to 100, how well does this candidate's resume match the job description? Provide a JSON response with:
    - "score": Numeric score (0–100).
    - "justification": Brief explanation (2–3 sentences).
    - "skills_present": List of matched skills.
    - "skills_missing": List of missing skills.
    Ensure the response is valid JSON.
    """
    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": 1000,  # Increased from 500
                    "temperature": 0.5,
                    "topP": 0.9
                }
            })
        )
        raw_response = response['body'].read().decode()
        debug_logs['titan_response'] = f"Titan raw response: {raw_response[:500]}..."
        result = json.loads(raw_response)['results'][0]['outputText']
        debug_logs['titan_output'] = f"Titan output text: {result[:500]}..."
        try:
            feedback = json.loads(result)
        except json.JSONDecodeError as e:
            debug_logs['titan_json_error'] = f"JSON parsing failed: {str(e)}. Attempting to fix..."
            # Attempt to fix unterminated string
            result = result[:result.rfind('}') + 1]  # Truncate to last valid JSON
            feedback = json.loads(result)
        with open(cache_file, 'w') as f:
            json.dump(feedback, f)
        debug_logs['titan_cache_saved'] = f"Saved Titan feedback to cache: {cache_file}"
        return feedback
    except Exception as e:
        debug_logs['titan_error'] = f"Titan feedback failed: {str(e)}"
        st.error(f"❌ Titan feedback failed: {str(e)}")
        return None

def process_sqs_queue(queue_url, debug_logs=None):
    sqs = boto3.client(
        'sqs',
        aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    try:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20
        )
        debug_logs['sqs_receive'] = f"SQS receive response: {json.dumps(response, indent=2)[:200]}..."
        if 'Messages' in response:
            message = json.loads(response['Messages'][0]['Body'])
            receipt_handle = response['Messages'][0]['ReceiptHandle']
            debug_logs['sqs_message'] = f"Processing SQS message: {json.dumps(message, indent=2)[:200]}..."
            feedback = get_titan_feedback(message['jd_text'], message['cv_text'], debug_logs=debug_logs)
            if feedback:
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle
                )
                debug_logs['sqs_deleted'] = "SQS message deleted"
            return feedback
        debug_logs['sqs_empty'] = "No messages in SQS queue"
        st.warning("No messages in SQS queue")
        return None
    except Exception as e:
        debug_logs['sqs_process_error'] = f"Failed to process SQS queue: {str(e)}"
        st.error(f"❌ Failed to process SQS queue: {str(e)}")
        return None

def compute_file_hash(file_content):
    return hashlib.md5(file_content).hexdigest()

def get_embedding(text, debug_logs=None):
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
            debug_logs['embed_cache'] = f"Loaded embedding from cache: {cache_file}"
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
        debug_logs['embed_saved'] = f"Saved embedding to cache: {cache_file}"
        return embedding
    except Exception as e:
        debug_logs['embed_error'] = f"Error generating embedding: {str(e)}"
        st.error(f"Error generating embedding: {str(e)}")
        return None

def extract_resume_skills_section(resume_text, debug_logs=None):
    skills_pattern = r'(?i)(?:Skills|Technical Skills|Key Skills|Proficiencies|Competencies|Technologies|Tools|Expertise)[\s:]*\n([\s\S]*?)(?=\n\n[A-Z]|$)'
    match = re.search(skills_pattern, resume_text)
    if match:
        debug_logs['skills_section'] = f"Extracted skills section: {match.group(1)[:200]}..."
        return match.group(1).strip()
    keywords = r'(?i)\b(Python|NumPy|Pandas|Scikit-learn|TensorFlow|PyTorch|Git|AWS|GCP|Azure|Spark|Hadoop|SQL|NoSQL|Docker|MLOps|Machine Learning|Deep Learning|Statistical Analysis|Data Preprocessing|Feature Engineering|Model Evaluation|Communication|Problem Solving|Teamwork|Agile|Neural Networks|Probability Theory)\b'
    matches = re.finditer(keywords, resume_text)
    skills_text = []
    for match in matches:
        start = max(0, resume_text.rfind('\n', 0, match.start()))
        end = resume_text.find('\n', match.end())
        if end == -1:
            end = len(resume_text)
        skills_text.append(resume_text[start:end].strip())
    if skills_text:
        debug_logs['skills_keywords'] = f"Extracted skills from keywords: {', '.join(skills_text[:3])}..."
        return '\n'.join(skills_text)
    project_pattern = r'(?i)(?:Projects|Project Experience)[\s:]*\n([\s\S]*?)(?=\n\n[A-Z]|$)'
    match = re.search(project_pattern, resume_text)
    if match:
        debug_logs['skills_projects'] = f"Extracted skills from projects: {match.group(1)[:200]}..."
        return match.group(1).strip()
    debug_logs['skills_fallback'] = "No skills section found, using full resume text"
    return resume_text

def preprocess_skill(skill, debug_logs=None):
    key_terms = re.findall(
        r'\b(Python|NumPy|Pandas|Scikit-learn|TensorFlow|PyTorch|Git|AWS|GCP|Azure|Spark|Hadoop|SQL|NoSQL|Docker|MLOps|Machine Learning|Deep Learning|Statistical Analysis|Data Preprocessing|Feature Engineering|Model Evaluation|Communication|Problem Solving|Teamwork|Agile|Neural Networks|Probability Theory)\b',
        skill, re.I
    )
    if key_terms:
        debug_logs['skill_terms'] = f"Preprocessed skill '{skill}' to terms: {key_terms}"
        return key_terms
    debug_logs['skill_fallback'] = f"No key terms for skill '{skill}', using first two words"
    return [' '.join(skill.split()[:2])]

def compute_skill_matches(required_skills, resume_text, debug_logs=None):
    if not required_skills:
        debug_logs['skill_match_none'] = "No required skills provided"
        return [], []
    resume_skills_text = extract_resume_skills_section(resume_text, debug_logs)
    skills_present = []
    skills_missing = []
    resume_embedding = get_embedding(resume_skills_text, debug_logs)
    if resume_embedding is None:
        debug_logs['skill_match_error'] = "Failed to generate resume embedding"
        return [], required_skills
    
    for skill in required_skills:
        skill_terms = preprocess_skill(skill, debug_logs)
        max_similarity = 0.0
        best_term = ''
        for term in skill_terms:
            skill_embedding = get_embedding(term, debug_logs)
            if skill_embedding is None:
                continue
            similarity = cosine_similarity([skill_embedding], [resume_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_term = term
        debug_logs[f"skill_similarity_{skill}"] = f"Skill '{skill}' (term: '{best_term}') cosine similarity: {max_similarity:.3f}"
        if max_similarity > 0.15:
            skills_present.append(skill)
        else:
            skills_missing.append(skill)
    debug_logs['skills_present'] = f"Skills Present: {skills_present}"
    debug_logs['skills_missing'] = f"Skills Missing: {skills_missing}"
    return skills_present, skills_missing

def get_dynamodb_records(debug_logs=None):
    dynamodb = boto3.client(
        'dynamodb',
        aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
        region_name='us-east-1'
    )
    try:
        response = dynamodb.scan(TableName='ResumeEvaluations')
        debug_logs['dynamodb_response'] = f"DynamoDB scan returned {len(response.get('Items', []))} items"
        debug_logs['dynamodb_raw'] = f"Raw DynamoDB response: {json.dumps(response, indent=2)[:200]}..."
        records = []
        for item in response.get('Items', []):
            record = {
                'ResumeID': item.get('ResumeID', {}).get('S', ''),
                'Name': item.get('Name', {}).get('S', 'Unknown'),
                'Email': item.get('Email', {}).get('S', 'Unknown'),
                'JobRole': item.get('JobRole', {}).get('S', 'Unknown'),
                'FinalScore': float(item.get('FinalScore', {}).get('N', 0)),
                'ATS_Score': float(item.get('ATS_Score', {}).get('N', 0)), 
                'TitanScore': float(item.get('TitanScore', {}).get('N', 0)),
                'Timestamp': item.get('Timestamp', {}).get('S', ''),
                'TitanJustification': item.get('TitanJustification', {}).get('S', ''),
                'SkillsPresent': item.get('SkillsPresent', {}).get('SS', item.get('PresentSkills', {}).get('SS', [])),
                'SkillsMissing': item.get('SkillsMissing', {}).get('SS', item.get('MissingSkills', {}).get('SS', []))
            }
            if isinstance(record['SkillsPresent'], str):
                try:
                    record['SkillsPresent'] = json.loads(record['SkillsPresent'])
                except:
                    record['SkillsPresent'] = []
            if isinstance(record['SkillsMissing'], str):
                try:
                    record['SkillsMissing'] = json.loads(record['SkillsMissing'])
                except:
                    record['SkillsMissing'] = []
            records.append(record)
        return records
    except Exception as e:
        debug_logs['dynamodb_error'] = f"Failed to fetch DynamoDB records: {str(e)}"
        st.error(f"❌ Failed to fetch DynamoDB records: {str(e)}")
        return []

def admin_dashboard(debug_logs=None):
    st.header("Admin Dashboard")
    password = st.text_input("Enter Admin Password", type="password")
    if password != "admin123":
        st.error("❌ Incorrect password")
        return
    
    st.subheader("Evaluation Records")
    records = get_dynamodb_records(debug_logs)
    if not records:
        st.warning("No records found in DynamoDB")
        return
    
    df = pd.DataFrame(records)
    st.write(f"**Total Evaluations**: {len(records)}")
    st.write(f"**Average Final Score**: {df['FinalScore'].mean():.2f}%")
    st.write(f"**Top Job Roles**: {', '.join(df['JobRole'].value_counts().head(3).index)}")

    search_term = st.text_input("Search by Name, Email, or Job Role")
    if search_term:
        df = df[
            df['Name'].str.contains(search_term, case=False, na=False) |
            df['Email'].str.contains(search_term, case=False, na=False) |
            df['JobRole'].str.contains(search_term, case=False, na=False)
        ]
    
    st.dataframe(
        df[['ResumeID', 'Name', 'Email', 'JobRole', 'FinalScore', 'TitanScore', 'Timestamp']],
        use_container_width=True
    )

    selected_id = st.selectbox("Select ResumeID for Details", df['ResumeID'])
    if selected_id:
        record = df[df['ResumeID'] == selected_id].iloc[0]
        st.subheader("Record Details")
        st.write(f"**Name**: {record['Name']}")
        st.write(f"**Email**: {record['Email']}")
        st.write(f"**Job Role**: {record['JobRole']}")
        st.write(f"**Final Score**: {record['FinalScore']:.2f}%")
        st.write(f"**Titan Score**: {record['TitanScore']:.2f}%")
        st.write(f"**Titan Justification**: {record['TitanJustification']}")
        st.write(f"**Skills Present**: {', '.join(record['SkillsPresent'])}")
        st.write(f"**Skills Missing**: {', '.join(record['SkillsMissing'])}")
        timestamp = record['Timestamp'].replace(':', '-').split('.')[0]
        resume_url = f"https://{config('S3_BUCKET_NAME')}.s3.amazonaws.com/resumes/resume_{record['Name'].replace(' ', '_')}_{timestamp}.pdf"
        report_url = f"https://{config('S3_BUCKET_NAME')}.s3.amazonaws.com/reports/report_{record['Name'].replace(' ', '_')}_{timestamp}.txt"
        st.write(f"[View Resume on S3]({resume_url})")
        st.write(f"[View Report on S3]({report_url})")

def main():
    st.title("AI-Powered Resume Evaluator")
    debug_logs = {}
    tabs = st.tabs(["Resume Evaluator", "Admin Dashboard"])
    
    with tabs[0]:
        st.info("ℹ️ Scroll to the end to view detailed feedback after processing.")
        st.header("Resume Upload and Job Role")
        jd_option = st.radio("Job Description Input", ["Upload JD", "Enter Job Role"])
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        if st.button("Upload Resume"):
            if uploaded_file:
                st.success(f"Resume uploaded: {uploaded_file.name}")
            else:
                st.error("Please select a PDF file to upload.")
        
        if jd_option == "Upload JD":
            uploaded_jd = st.file_uploader("Upload Job Description (Text)", type=["txt"])
            if st.button("Upload JD"):
                if uploaded_jd:
                    st.success(f"JD uploaded: {uploaded_jd.name}")
                else:
                    st.error("Please select a TXT file to upload.")
        else:
            uploaded_jd = None
        
        job_role = st.text_input("Enter Job Role (e.g., Machine Learning Engineer, Porter)", placeholder="Type job role here")
        
        queue_url = config('SQS_QUEUE_URL', default='https://sqs.us-east-1.amazonaws.com/<account_id>/ResumeFeedbackQueue')
        debug_logs['queue_url'] = f"SQS Queue URL: {queue_url}"

        if st.button("Process Resume"):
            if not uploaded_file:
                st.error("❌ Please upload a resume.")
                st.stop()
            if not job_role and jd_option == "Enter Job Role":
                st.error("❌ Please enter a job role.")
                st.stop()
            if jd_option == "Upload JD" and not uploaded_jd:
                st.error("❌ Please upload a job description.")
                st.stop()

            file_content = uploaded_file.read()
            resume_hash = compute_file_hash(file_content)
            cache_file = f"{CACHE_DIR}/resume_{resume_hash}.txt"
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    resume_text = f.read()
                debug_logs['resume_cache'] = f"Loaded resume text from cache: {cache_file}"
                st.info("⏳ Loaded resume text from cache...")
            else:
                with open("temp_resume.pdf", "wb") as f:
                    f.write(file_content)
                st.info("⏳ Extracting text from resume using AWS Textract...")
                try:
                    resume_text = extract_text_from_pdf("temp_resume.pdf", config('AWS_ACCESS_KEY_ID'), config('AWS_SECRET_ACCESS_KEY'))
                    with open(cache_file, 'w') as f:
                        f.write(resume_text)
                    debug_logs['resume_extracted'] = f"Saved resume text to cache: {cache_file}"
                except Exception as e:
                    debug_logs['textract_error'] = f"Failed to extract text from resume: {str(e)}"
                    st.error(f"❌ Failed to extract text from resume: {str(e)}")
                    st.stop()
            
            debug_logs['resume_text'] = f"Extracted Resume Text (first 200 chars): {resume_text[:200]}"
            debug_logs['resume_length'] = f"Resume Text Length: {len(resume_text)}"

            st.header("Resume Details")
            details = extract_details(resume_text)
            st.write(f"**Name**: {details['Name']}")
            st.write(f"**Email**: {details['Email']}")
            st.write(f"**Projects**: {', '.join(details['Projects'])}")
            debug_logs['resume_details'] = f"Extracted Details: Name={details['Name']}, Email={details['Email']}, Projects={details['Projects']}"

            st.header("Job Description Details")
            jd_text = ""
            required_skills = []
            if jd_option == "Upload JD" and uploaded_jd:
                try:
                    jd_buffer = io.BytesIO(uploaded_jd.read())
                    jd_text = jd_buffer.read().decode("utf-8", errors="ignore")
                    debug_logs['jd_uploaded'] = f"Extracted JD Text (first 200 chars): {jd_text[:200]}"
                    debug_logs['jd_length'] = f"JD Text Length: {len(jd_text)}"
                    required_skills = re.findall(r'\b[A-Za-z\s\-]+(?:skills|experience|proficiency)\b', jd_text, re.IGNORECASE)
                    required_skills = [s.replace('skills', '').replace('experience', '').replace('proficiency', '').strip() for s in required_skills]
                    debug_logs['jd_skills'] = f"Extracted JD Skills: {required_skills}"
                except Exception as e:
                    debug_logs['jd_upload_error'] = f"Failed to read job description: {str(e)}"
                    st.error(f"❌ Failed to read job description: {str(e)}")
                    jd_text = job_role
                    required_skills = []
            else:
                st.info("⏳ Generating job description with Bedrock...")
                jd_data = generate_jd_with_bedrock(job_role, debug_logs=debug_logs)
                jd_text = jd_data["job_description"]
                required_skills = jd_data["skills"]
                debug_logs['jd_generated'] = f"Generated JD Text (first 200 chars): {jd_text[:200]}"
                debug_logs['jd_length'] = f"JD Text Length: {len(jd_text)}"
                debug_logs['jd_skills'] = f"Bedrock Skills: {required_skills}"
            
            if not jd_text.strip():
                jd_text = job_role
                required_skills = []
            
            st.write(f"**Job Role**: {job_role}")
            st.write(f"**Skills Required**: {', '.join(required_skills)}")

            debug_logs['required_skillsmid'] = f"Required Skills: {required_skills}"
            skills_present, skills_missing = compute_skill_matches(required_skills, resume_text, debug_logs)
            debug_logs['skill_matches'] = f"Skill Matches: {len(skills_present)}/{len(required_skills)}"

            st.header("Evaluation Results")
            try:
                ats_results = calculate_ats_score(resume_text, jd_text)
                ats_score = ats_results['score']
                resume_tokens = ats_results['resume_tokens']
                jd_tokens = ats_results['jd_tokens']
                tfidf_matrix_shape = ats_results['tfidf_matrix_shape']
                cosine_sim = ats_results['cosine_sim']
                feature_names = ats_results['feature_names']
                debug_info = ats_results.get('debug_info', {})
                for key, value in debug_info.items():
                    debug_logs[f"ats_{key}"] = value
            except Exception as e:
                debug_logs['ats_error'] = f"Failed to calculate ATS score: {str(e)}"
                st.error(f"❌ Failed to calculate ATS score: {str(e)}")
                st.stop()
            skill_matches = len(skills_present) / len(required_skills) if required_skills else 0
            skill_bonus = skill_matches * 0.4
            ats_final_score = min(ats_score + skill_bonus * 100, 100.0)
            debug_logs['ats_score'] = f"ATS Score (Internal): {ats_final_score:.2f}%"
            debug_logs['embedding_score'] = f"Embedding Score: {ats_score}"
            debug_logs['resume_tokens'] = f"Resume Tokens (first 20): {resume_tokens[:20]}"
            debug_logs['jd_tokens'] = f"JD Tokens (first 20): {jd_tokens[:20]}"
            debug_logs['matrix_shape'] = f"Embedding Matrix Shape: {tfidf_matrix_shape}"
            debug_logs['cosine_sim'] = f"Cosine Similarity: {cosine_sim}"
            debug_logs['skill_match_ratio'] = f"Skill Matches: {skill_matches}"
            debug_logs['skill_bonus'] = f"Skill Bonus: {skill_bonus}"

            # SQS and Feedback
            st.info("⏳ Sending feedback request to SQS...")
            message_id = send_to_sqs(jd_text, resume_text, queue_url, debug_logs)
            feedback_result = None
            score_type = "ATS only"
            if message_id:
                st.info("⏳ Processing feedback from SQS queue...")
                for _ in range(5):
                    feedback_result = process_sqs_queue(queue_url, debug_logs)
                    if feedback_result:
                        st.session_state.feedback = feedback_result
                        score_type = "60% ATS + 40% AI Feedback"  # Updated
                        debug_logs['score_type'] = f"Score Type: {score_type}"
                        break
                    time.sleep(5)
                if not feedback_result:
                    debug_logs['sqs_feedback_error'] = "Failed to generate feedback from SQS"
                    st.error("❌ Failed to generate feedback from SQS.")

            # Final Score
            if feedback_result:
                final_score = 0.6 * ats_final_score + 0.4 * feedback_result['score']
                final_score = min(final_score, 100.0)
                st.write(f"**Final Score**: {final_score:.2f}% ({score_type})")
                debug_logs['final_score'] = f"Final Score: {final_score:.2f}% ({score_type})"
            else:
                final_score = ats_final_score
                st.write(f"**Final Score**: {final_score:.2f}% ({score_type})")
                debug_logs['final_score'] = f"Final Score: {final_score:.2f}% ({score_type})"

            debug_logs['ats_final_score'] = f"ATS Final Score: {ats_final_score:.2f}%"
            st.session_state.final_score = final_score
            st.session_state.score_type = score_type

            with st.expander("View Debug Logs"):
                st.json(debug_logs)

            resume_url = ""
            report_url = ""
            try:
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                resume_url = upload_to_s3(
                    "temp_resume.pdf",
                    config('S3_BUCKET_NAME'),
                    f"resumes/resume_{details['Name'].replace(' ', '_')}_{timestamp}.pdf",
                    config('AWS_ACCESS_KEY_ID'),
                    config('AWS_SECRET_ACCESS_KEY')
                )
                report = f"Resume Evaluation Report\nName: {details['Name']}\nEmail: {details['Email']}\nJob Role: {job_role}\nFinal Score: {final_score:.2f}%\nSkills Present: {', '.join(skills_present)}\nSkills Missing: {', '.join(skills_missing)}"
                if feedback_result:
                    report += f"\nAI Feedback:\nJustification: {feedback_result['justification']}\nSkills Present: {', '.join(feedback_result['skills_present'])}\nSkills Missing: {', '.join(feedback_result['skills_missing'])}"
                with open("temp_report.txt", "w") as f:
                    f.write(report)
                report_url = upload_to_s3(
                    "temp_report.txt",
                    config('S3_BUCKET_NAME'),
                    f"reports/report_{details['Name'].replace(' ', '_')}_{timestamp}.txt",
                    config('AWS_ACCESS_KEY_ID'),
                    config('AWS_SECRET_ACCESS_KEY')
                )
                st.write(f"[View Resume on S3]({resume_url})")
                st.write(f"[View Report on S3]({report_url})")
                debug_logs['s3_resume'] = f"Uploaded resume to: {resume_url}"
                debug_logs['s3_report'] = f"Uploaded report to: {report_url}"
            except Exception as e:
                debug_logs['s3_error'] = f"Failed to upload to S3: {str(e)}"
                st.error(f"❌ Failed to upload to S3: {str(e)}")

            dynamodb = boto3.client(
                'dynamodb',
                aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
                region_name='us-east-1'
            )
            try:
                item = {
                    'ResumeID': {'S': str(uuid.uuid4())},
                    'Name': {'S': details['Name']},
                    'Email': {'S': details['Email']},
                    'JobRole': {'S': job_role},
                    'FinalScore': {'N': str(final_score)},
                    'ATS_Score': {'N': str(ats_final_score)},  # Added
                    'Timestamp': {'S': datetime.datetime.now().isoformat()},
                    'SkillsPresent': {'SS': skills_present if skills_present else ['None']},
                    'SkillsMissing': {'SS': skills_missing if skills_missing else ['None']}
                }
                if feedback_result:
                    item['TitanScore'] = {'N': str(feedback_result['score'])}
                    item['TitanJustification'] = {'S': feedback_result['justification']}
                debug_logs['dynamodb_item'] = f"DynamoDB Item: {item}"
                dynamodb.put_item(
                    TableName='ResumeEvaluations',
                    Item=item
                )
                st.success("✅ Evaluation results saved to DynamoDB.")
                debug_logs['dynamodb_success'] = "Evaluation results saved to DynamoDB"
            except Exception as e:
                debug_logs['dynamodb_error'] = f"Failed to save to DynamoDB: {str(e)}"
                st.error(f"❌ Failed to save to DynamoDB: {str(e)}")

            sns = boto3.client(
                'sns',
                aws_access_key_id=config('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=config('AWS_SECRET_ACCESS_KEY'),
                region_name='us-east-1'
            )
            try:
                message = f"Resume Evaluation Report\nName: {details['Name']}\nEmail: {details['Email']}\nJob Role: {job_role}\nFinal Score: {final_score:.2f}%\nSkills Present: {', '.join(skills_present)}\nSkills Missing: {', '.join(skills_missing)}"
                if feedback_result:
                    message += f"\nAI Feedback:\nJustification: {feedback_result['justification']}"
                sns.publish(
                    TopicArn=config('SNS_TOPIC_ARN'),
                    Message=message
                )
                st.success("✅ SNS notification sent.")
                debug_logs['sns_success'] = "SNS notification sent"
            except Exception as e:
                debug_logs['sns_error'] = f"Failed to send SNS notification: {str(e)}"
                st.error(f"❌ Failed to send SNS notification: {str(e)}")

        st.header("Feedback")
        if st.button("Show Feedback") and 'feedback' in st.session_state:
            feedback = st.session_state.feedback
            st.subheader("AI Feedback")
            final_score = st.session_state.get('final_score', 0)
            score_type = st.session_state.get('score_type', 'ATS only')
            st.write(f"**Final Score**: {final_score:.2f}% ({score_type})")
            st.write(f"**Justification**: {feedback['justification']}")
            st.write(f"**Skills Present**: {', '.join(feedback['skills_present'])}")
            st.write(f"**Skills Missing**: {', '.join(feedback['skills_missing'])}")

    with tabs[1]:
        admin_dashboard(debug_logs)

if __name__ == "__main__":
    main()