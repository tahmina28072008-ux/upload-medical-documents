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

CORS(app, origins=["https://healthcare-patient-portal.web.app"])

BUCKET_NAME = "upload-documents-report"
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}

GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)
else:
    raise Exception("GEMINI_API_KEY environment variable not set.")

CANDIDATE_MODELS = [
    'models/gemini-2.5-flash',
    'models/gemini-2.5-flash-preview-05-20',
    'models/gemini-1.5-flash',
    'models/gemini-1.5-pro'
]
AVAILABLE_MODELS = [m.name for m in genai.list_models()]
DEFAULT_MODEL = next((m for m in CANDIDATE_MODELS if m in AVAILABLE_MODELS), "models/gemini-2.5-flash")
print(f"Using Gemini model: {DEFAULT_MODEL}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def upload_to_gcs(file_obj, filename):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_file(file_obj, content_type=file_obj.content_type)
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"


def extract_text_from_pdf_bytes(pdf_bytes):
    print("Starting PDF extraction...")
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages[:5] if page.extract_text()])
    print(f"Extracted text from PDF ({len(text)} chars)")
    return text


def summarize_with_gemini(prompt: str, model_name=DEFAULT_MODEL):
    print("Sending prompt to Gemini:", prompt[:200])
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt])
        print("Gemini raw response:", response)
        if hasattr(response, "candidates") and response.candidates:
            summary = response.candidates[0].content.parts[0].text.strip()
        else:
            summary = ""
        print("Gemini summary:", summary[:200])
        return summary
    except Exception as e:
        print("Gemini error:", str(e))
        return f"Error processing the report: {str(e)}"


def ai_summarize(report_content):
    # Limit content for Gemini to avoid crashes/timeouts
    report_content = report_content[:1000]
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


@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload route hit")
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{int(round(os.times()[4]*1000))}_{filename}"
        public_url = upload_to_gcs(file, unique_filename)
        print(f"Returning fileUrl: {public_url}")
        return jsonify({'fileUrl': public_url})
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/webhook', methods=['POST'])
def webhook():
    print("Webhook route hit")
    body = request.json
    print(f"Webhook request body: {body}")
    params = body.get('sessionInfo', {}).get('parameters', {})
    file_url = params.get('file_url')

    patient_summary = "No file URL provided."
    doctor_summary = "No file URL provided."

    if file_url:
        try:
            print("Fetching file from:", file_url)
            response = requests.get(file_url, timeout=15)
            print("Report fetch status:", response.status_code)

            if response.status_code == 200:
                if file_url.lower().endswith('.pdf'):
                    report_content = extract_text_from_pdf_bytes(response.content)
                else:
                    report_content = response.content.decode('utf-8', errors='ignore')

                print(f"Extracted report_content: {repr(report_content[:200])}")

                if not report_content.strip():
                    patient_summary = doctor_summary = "The report file appears empty or could not be read."
                else:
                    patient_summary, doctor_summary = ai_summarize(report_content)

                print("✅ Patient summary generated")
                print(patient_summary[:300])
                print("✅ Doctor summary generated")
                print(doctor_summary[:300])

            else:
                patient_summary = doctor_summary = (
                    f"Could not retrieve the report file (HTTP {response.status_code}). Please try uploading again."
                )

        except Exception as e:
            print("Error processing file:", str(e))
            patient_summary = doctor_summary = f"Error processing the report: {str(e)}"

    # Debug logs for what we send back to Dialogflow
    print("Returning to Dialogflow with session parameters:")
    print({
        "patient_summary": patient_summary[:200],
        "doctor_summary": doctor_summary[:200]
    })

    return jsonify({
        "sessionInfo": {
            "parameters": {
                "patient_summary": patient_summary,
                "doctor_summary": doctor_summary
            }
        },
        "fulfillment_response": {
            "messages": [
                {
                    "text": {
                        "text": [
                            f"Here’s what I found in your report:\n\n"
                            f"🧾 Patient summary:\n{patient_summary}\n\n"
                            f"👨‍⚕️ Doctor summary:\n{doctor_summary}\n\n"
                            f"Do you want to confirm this booking?"
                        ]
                    }
                }
            ]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
