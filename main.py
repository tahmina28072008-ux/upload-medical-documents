import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google.cloud import storage
import google.generativeai as genai
import requests
import pdfplumber

app = Flask(__name__)

# Allow requests from your frontend domain
CORS(app, origins=["https://healthcare-patient-portal.web.app"])

BUCKET_NAME = "upload-documents-report"
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}

# Configure Gemini API key from environment variable
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
    # Do NOT call blob.make_public(); use bucket-level IAM for access.
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"


def extract_text_from_pdf_bytes(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])


def summarize_with_gemini(prompt: str, model_name="gemini-1.5"):
    """Helper to summarize text with Gemini."""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt])
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        return f"Error processing the report: {str(e)}"


def ai_summarize(report_content):
    # Patient summary prompt
    prompt_patient = (
        "Summarize the following medical report for a patient in simple, natural language. "
        "Focus on what the patient needs to know, avoid medical jargon, and explain clearly:\n\n"
        f"{report_content}"
    )
    # Doctor summary prompt
    prompt_doctor = (
        "Summarize the following medical report for a doctor, highlighting key findings, clinical concerns, "
        "and what to focus on before the patient's visit:\n\n"
        f"{report_content}"
    )

    patient_summary = summarize_with_gemini(prompt_patient)
    doctor_summary = summarize_with_gemini(prompt_doctor)

    return patient_summary, doctor_summary


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
    patient_summary = "No file URL provided."
    doctor_summary = "No file URL provided."
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
                    patient_summary = doctor_summary = "The report file appears empty or could not be read."
                else:
                    patient_summary, doctor_summary = ai_summarize(report_content)
            else:
                patient_summary = doctor_summary = (
                    f"Could not retrieve the report file (HTTP {response.status_code}). Please try uploading again."
                )
        except Exception as e:
            patient_summary = doctor_summary = f"Error processing the report: {str(e)}"
    return jsonify({
        "fulfillment_response": {
            "messages": [
                {"text": {"text": [f"Patient summary: {patient_summary}\n\nDoctor summary: {doctor_summary}"]}}
            ]
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
