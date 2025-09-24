from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google.cloud import storage
import os

app = Flask(__name__)

# Allow requests from your frontend domain
CORS(app, origins=["https://healthcare-patient-portal.web.app"])

BUCKET_NAME = "upload-documents-report"
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_gcs(file_obj, filename):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_file(file_obj, content_type=file_obj.content_type)
    # Do NOT call blob.make_public(); use bucket-level IAM for access.
    # Return the object's URL (works if bucket is public, otherwise use signed URL)
    return f"https://storage.googleapis.com/{BUCKET_NAME}/{filename}"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Use a unique name, e.g. timestamp + filename
        unique_filename = f"{int(round(os.times()[4]*1000))}_{filename}"
        public_url = upload_to_gcs(file, unique_filename)
        return jsonify({'fileUrl': public_url})
    return jsonify({'error': 'Invalid file type'}), 400

# Dialogflow CX webhook fulfillment endpoint
@app.route('/webhook', methods=['POST'])
def webhook():
    import requests  # Local import for Cloud Run compatibility
    body = request.json
    file_url = body['sessionInfo']['parameters'].get('file_url')
    summary = "No file URL provided."
    if file_url:
        try:
            response = requests.get(file_url)
            if response.status_code == 200:
                report_content = response.content.decode('utf-8', errors='ignore')
                summary = analyze_medical_report(report_content)
            else:
                summary = f"Could not retrieve the report file (HTTP {response.status_code}). Please try uploading again."
        except Exception as e:
            summary = f"Error processing the report: {str(e)}"
    return jsonify({
        "fulfillment_response": {
            "messages": [
                {"text": {"text": [f"Here is your report summary: {summary}"]}}
            ]
        }
    })

def analyze_medical_report(content):
    # Placeholder: Replace with your own analysis logic
    return content[:500] + ("..." if len(content) > 500 else "")

if __name__ == '__main__':
    app.run(debug=True)
