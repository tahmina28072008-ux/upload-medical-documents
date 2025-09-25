import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google.cloud import storage
import google.generativeai as genai
import requests
import pdfplumber
import re

app = Flask(__name__)

CORS(app, origins=["https://healthcare-patient-portal.web.app"])

BUCKET_NAME = "upload-documents-report"
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    raise Exception("GEMINI_API_KEY environment variable not set.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_gcs(file_obj, filename):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_file(file_obj, content_type=file_obj.content_type)
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"

def extract_text_from_pdf_bytes(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def summarize_with_gemini(prompt: str, model_name="gemini-1.5-flash"):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt])
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        return f"Error processing the report: {str(e)}"

def ai_summarize(report_content):
    prompt_patient = (
        "Summarize the following medical report for a patient in simple, natural language. "
        "Focus on what the patient needs to know, avoid medical jargon, and explain clearly:\n\n"
        f"{report_content}"
    )
    prompt_doctor = (
        "Summarize the following medical report for a doctor, highlighting key findings, clinical concerns, "
        "and what to focus on before the patient's visit:\n\n"
        f"{report_content}"
    )
    patient_summary = summarize_with_gemini(prompt_patient)
    doctor_summary = summarize_with_gemini(prompt_doctor)
    return patient_summary, doctor_summary

# --- Structuring Helpers ---

def parse_patient_summary(raw_text):
    sentences = [s.strip() for s in re.split(r"\.\s+", raw_text) if s.strip()]
    intro = sentences[0] if sentences else ""
    findings = []
    recommendations = []
    for s in sentences[1:]:
        if "recommend" in s.lower():
            recommendations.append(s)
        elif (
            "no signs" in s.lower()
            or "normal" in s.lower()
            or "nothing to worry" in s.lower()
            or "looks good" in s.lower()
            or "no problems" in s.lower()
        ):
            findings.append(s)
        else:
            findings.append(s)
    if not findings and len(sentences) > 1:
        findings = sentences[1:]
    return {
        "intro": intro,
        "findings": findings,
        "recommendations": recommendations,
    }

def parse_doctor_summary(raw_text):
    patient_match = re.search(r'Patient ([^(]+\(.*?\))', raw_text)
    test_match = re.search(r'underwent ([^.]+)\.', raw_text)
    findings_match = re.search(r'All results, ([^.]+)\.', raw_text)
    recommendation_match = re.search(r'recommendation is ([^.]+)\.', raw_text)
    clinical_match = re.search(r'(No immediate clinical concerns[^.]*\.)', raw_text)
    prep_match = re.search(r'Before the visit, ([^.]+)\.', raw_text)

    findings = []
    if findings_match:
        findings = [f.strip() for f in re.split(', | and ', findings_match.group(1))]
    else:
        fallback = re.findall(r'([^.]+normal limits)', raw_text)
        findings.extend([f.strip() for f in fallback])
        findings.extend(re.findall(r'(No signs of infection)', raw_text))

    recommendations = []
    if recommendation_match:
        recommendations.append(recommendation_match.group(1).strip())
    else:
        rec_match = re.search(r'Dr\. Carter[^\n\.]*recommend[^\n\.]*', raw_text)
        if rec_match:
            recommendations.append(rec_match.group(0).strip())

    return {
        "patient": patient_match.group(1).strip() if patient_match else None,
        "test": test_match.group(1).strip() if test_match else None,
        "findings": findings,
        "recommendations": recommendations,
        "clinical": clinical_match.group(1).strip() if clinical_match else None,
        "preparation": prep_match.group(1).strip() if prep_match else None,
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{int(round(os.times()[4]*1000))}_{filename}"
        public_url = upload_to_gcs(file, unique_filename)
        return jsonify({'fileUrl': public_url})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/webhook', methods=['POST'])
def webhook():
    body = request.json
    file_url = body['sessionInfo']['parameters'].get('file_url')
    patient_structured = {}
    doctor_structured = {}
    if file_url:
        try:
            response = requests.get(file_url)
            if response.status_code == 200:
                # Detect file type from URL extension
                if file_url.lower().endswith('.pdf'):
                    report_content = extract_text_from_pdf_bytes(response.content)
                else:
                    report_content = response.content.decode('utf-8', errors='ignore')
                if not report_content.strip():
                    patient_structured = {"intro": "The report file appears empty or could not be read."}
                    doctor_structured = {"patient": None, "test": None, "findings": [], "recommendations": [], "clinical": None, "preparation": None}
                else:
                    patient_summary, doctor_summary = ai_summarize(report_content)
                    patient_structured = parse_patient_summary(patient_summary)
                    doctor_structured = parse_doctor_summary(doctor_summary)
            else:
                err_msg = f"Could not retrieve the report file (HTTP {response.status_code}). Please try uploading again."
                patient_structured = {"intro": err_msg}
                doctor_structured = {"patient": None, "test": None, "findings": [], "recommendations": [], "clinical": None, "preparation": None}
        except Exception as e:
            err_msg = f"Error processing the report: {str(e)}"
            patient_structured = {"intro": err_msg}
            doctor_structured = {"patient": None, "test": None, "findings": [], "recommendations": [], "clinical": None, "preparation": None}
    else:
        patient_structured = {"intro": "No file URL provided."}
        doctor_structured = {"patient": None, "test": None, "findings": [], "recommendations": [], "clinical": None, "preparation": None}

    return jsonify({
        "fulfillment_response": {
            "messages": [
                {
                    "payload": {
                        "patient_summary": patient_structured,
                        "doctor_summary": doctor_structured
                    }
                }
            ]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
