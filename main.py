import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import requests
from flask_cors import CORS  # <-- Added

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
# Allow only your frontend(s) for security; add more origins as needed
CORS(app, origins=[
    "http://healthcare-patient-portal.web.app",
    "https://healthcare-patient-portal.web.app"
])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# File upload endpoint
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{str(int(round(os.times()[4]*1000)))}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        file_url = request.host_url + 'uploads/' + unique_filename
        return jsonify({'fileUrl': file_url})
    return jsonify({'error': 'Invalid file type'}), 400

# Serve uploaded files
@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Dialogflow CX webhook fulfillment endpoint
@app.route('/webhook', methods=['POST'])
def webhook():
    body = request.json
    file_url = body['sessionInfo']['parameters'].get('file_url')
    summary = "No file URL provided."
    if file_url:
        try:
            response = requests.get(file_url)
            report_content = response.content.decode('utf-8', errors='ignore')
            summary = analyze_medical_report(report_content)
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
    # Replace this with your own AI/ML or logic
    return content[:500] + ("..." if len(content) > 500 else "")

if __name__ == '__main__':
    app.run(debug=True)
