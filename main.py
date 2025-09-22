import base64
import json
import os
import requests
from flask import Flask, request, jsonify
from google.cloud import firestore, storage, vision

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
# Environment variables for your bucket and project ID
GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME')

if not all([GOOGLE_CLOUD_PROJECT, GCS_BUCKET_NAME]):
    print("Warning: GOOGLE_CLOUD_PROJECT or GCS_BUCKET_NAME environment variables are not set.")

# Initialize Firestore, Storage, and Vision clients
db = firestore.Client(project=GOOGLE_CLOUD_PROJECT)
storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
vision_client = vision.ImageAnnotatorClient()

# --- Gemini API Configuration ---
# This URL is for gemini-2.5-flash-preview-05-20.
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
# Get the API key from the environment variable
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

@app.route('/', methods=['POST'])
def webhook():
    """
    Receives and processes the webhook request from Dialogflow CX.
    """
    try:
        # Get the request body from Dialogflow
        request_body = request.get_json(silent=True)
        print("Received webhook request:\n", json.dumps(request_body, indent=2))

        # Check for necessary parameters from Dialogflow's file upload event
        session_info = request_body.get('sessionInfo', {})
        parameters = session_info.get('parameters', {})
        document_url = parameters.get('document_url')
        document_name = parameters.get('document_name')

        if not document_url or not document_name:
            return jsonify({
                "fulfillmentResponse": {
                    "messages": [
                        {"text": {"text": ["I was not able to find the document. Please try uploading it again."]}}
                    ]
                }
            })

        print(f"Processing document: {document_name} from URL: {document_url}")

        # 1. Download the document from the temporary URL
        try:
            document_bytes = download_document(document_url)
            if not document_bytes:
                raise requests.exceptions.RequestException("Empty document content received.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading document: {e}")
            return jsonify({
                "fulfillmentResponse": {
                    "messages": [
                        {"text": {"text": ["I am sorry, I couldn't download the document. Please try uploading it again."]}}
                    ]
                }
            })

        # 2. Analyze the document with the Gemini API (via Vision API)
        gemini_response = analyze_document_with_gemini(document_bytes)
        if not gemini_response:
            return jsonify({
                "fulfillmentResponse": {
                    "messages": [
                        {"text": {"text": ["Sorry, I couldn't analyze the document. Please try again later."]}}
                    ]
                }
            })

        # 3. Store the original document in Google Cloud Storage
        original_document_url = save_document_to_storage(document_bytes, document_name)

        # 4. Store the analyzed data and document URL in Firestore
        session_id = request_body['session'].split('/')[-1]
        save_data_to_firestore(session_id, gemini_response, original_document_url)

        # 5. Send a success message back to Dialogflow
        return jsonify({
            "fulfillmentResponse": {
                "messages": [
                    {
                        "text": {
                            "text": [f"Thank you! I have successfully processed your medical document named '{document_name}'. The doctor can now review your test results prior to your appointment."]
                        }
                    }
                ]
            }
        })

    except Exception as e:
        print(f"An unexpected error occurred during webhook processing: {e}")
        return jsonify({
            "fulfillmentResponse": {
                "messages": [
                    {"text": {"text": [f"An unexpected error occurred. Please try again later."]}}
                ]
            }
        })

def download_document(url):
    """Downloads a document from a given URL."""
    response = requests.get(url)
    response.raise_for_status() # Raise an exception for bad status codes
    return response.content

def analyze_document_with_gemini(document_bytes):
    """
    Analyzes the document by first extracting text with the Vision API
    and then sending the text to the Gemini API for structured analysis.
    """
    try:
        # Step A: Extract text from the document using Google Cloud Vision API
        image = vision.Image(content=document_bytes)
        
        # This performs document text detection on the image file
        response = vision_client.document_text_detection(image=image)
        full_text = response.full_text_annotation.text
        
        if not full_text:
            print("Vision API did not extract any text from the document.")
            return None

        print("Extracted text from document:\n", full_text)

        # Step B: Send the extracted text to the Gemini API for analysis
        prompt = (
            "Analyze the following text from a medical document and extract the "
            "following information as a JSON object: 'patientName', 'dateOfBirth', "
            "'testName', 'results', 'doctorNotes', 'labName'. If a field is not found, "
            "use 'N/A'. The results should be a string or array of strings. "
            "Here is the text:\n\n"
            f"```\n{full_text}\n```"
        )
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "responseMimeType": "application/json"
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        gemini_response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
        gemini_response.raise_for_status()
        
        result = gemini_response.json()
        print("Gemini API response:\n", json.dumps(result, indent=2))
        
        # Extract and parse the JSON response from Gemini
        gemini_text = result['candidates'][0]['content']['parts'][0]['text']
        return json.loads(gemini_text)

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        return None
    except Exception as e:
        print(f"An error occurred during Vision/Gemini analysis: {e}")
        return None

def save_document_to_storage(document_bytes, document_name):
    """
    Saves the original document to Google Cloud Storage.
    Returns the public URL of the saved document.
    """
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        
        # Create a blob name with a unique identifier to prevent overwrites
        blob_path = f"dialogflow_uploads/{document_name}"

        # Upload the document
        blob = bucket.blob(blob_path)
        blob.upload_from_string(document_bytes)

        # Make the blob publicly readable
        blob.make_public()

        print(f"Document uploaded to {blob.public_url}")
        return blob.public_url

    except Exception as e:
        print(f"Error saving document to storage: {e}")
        return None

def save_data_to_firestore(session_id, data, document_url):
    """
    Saves the analyzed data and document URL to a Firestore document.
    """
    try:
        # Create a new document in a 'medical_history' collection,
        # using the session ID to link to the conversation.
        doc_ref = db.collection('medical_history').document(session_id)
        
        # Add a timestamp and the document URL to the data
        data['uploadTimestamp'] = firestore.SERVER_TIMESTAMP
        data['originalDocumentURL'] = document_url
        
        doc_ref.set(data)
        print(f"Data saved to Firestore under document ID: {doc_ref.id}")

    except Exception as e:
        print(f"Error saving data to Firestore: {e}")
        return None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
