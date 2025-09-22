import base64
import json
from flask import Flask, request, jsonify
import requests
from google.cloud import firestore
from google.cloud import storage

# Initialize Flask app
app = Flask(__name__)

# --- Configuration ---
# Replace with your own Google Cloud project ID
PROJECT_ID = "healthcare-patient-portal"

# Initialize Firestore and Storage clients
db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

# --- Gemini API Configuration ---
# You can use a specific Gemini model for document analysis.
# This URL is for gemini-2.5-flash-preview-05-20.
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"
# Replace with your Gemini API key (or leave empty for Canvas)
GEMINI_API_KEY = "AIzaSyBxGpoWp0kaztZDDe3eUulgZisEbzmIwPA"

@app.route('/', methods=['POST'])
def webhook():
    """
    Receives and processes the webhook request from Dialogflow CX.
    """
    try:
        # Get the request body from Dialogflow
        request_body = request.get_json(silent=True)
        print("Received webhook request:\n", json.dumps(request_body, indent=2))

        # Check for message and content (the uploaded file)
        if 'pageInfo' not in request_body or 'text' not in request_body['pageInfo']:
            return jsonify({"fulfillment_response": {"messages": [{"text": {"text": ["No file attachment found in the request."]}}]}})
            
        page_info = request_body['pageInfo']
        
        # Ensure 'text' field exists and is not empty before accessing
        if 'text' not in page_info:
            return jsonify({"fulfillment_response": {"messages": [{"text": {"text": ["No file attachment found in the request."]}}]}})

        request_text = page_info['text']
        
        # Extract the base64-encoded file data from the request.
        # Dialogflow sends the file as part of the 'text' field.
        # You will need a custom front-end to attach the file to the request.
        
        # This is a placeholder as Dialogflow's native integration may not send file data this way.
        # You will need to implement a front-end that sends a POST request with the base64 data.
        # For this example, we'll assume the base64 data is present in the request body.
        
        # In a real-world scenario, the front-end would send the document in a specific field.
        # This code assumes a simple structure for demonstration purposes.
        file_data_base64 = request_text
        file_bytes = base64.b64decode(file_data_base64)

        # 1. Analyze the document with the Gemini API
        gemini_response = analyze_document_with_gemini(file_bytes)
        if not gemini_response:
            return jsonify({"fulfillment_response": {"messages": [{"text": {"text": ["Sorry, I couldn't analyze the document. Please try again later."]}}]}})

        # 2. Store the original document in Google Cloud Storage
        original_document_url = save_document_to_storage(file_bytes)

        # 3. Store the analyzed data and document URL in Firestore
        patient_id = "patient_" + request_body['session'].split('/')[-1]
        save_data_to_firestore(patient_id, gemini_response, original_document_url)

        # 4. Send a success message back to Dialogflow
        return jsonify({
            "fulfillment_response": {
                "messages": [
                    {
                        "text": {
                            "text": [
                                f"Thank you! I have successfully processed your medical document. The doctor can now review your test results prior to your appointment."
                            ]
                        }
                    }
                ]
            }
        })

    except Exception as e:
        print(f"Error during webhook processing: {e}")
        return jsonify({"fulfillment_response": {"messages": [{"text": {"text": [f"An unexpected error occurred: {e}"]}}]}})

def analyze_document_with_gemini(document_bytes):
    """
    Sends the document bytes to the Gemini API for analysis.
    """
    try:
        payload = {
            "contents": {
                "role": "user",
                "parts": [
                    {
                        "text": "Analyze this medical document and extract the following information as a JSON object: 'patientName', 'dateOfBirth', 'testName', 'results', 'doctorNotes', 'labName'. If a field is not found, use 'N/A'. The results should be a string or array of strings."
                    },
                    {
                        "inlineData": {
                            "mimeType": "application/pdf",  # Or 'image/jpeg', 'image/png' etc.
                            "data": base64.b64encode(document_bytes).decode('utf-8')
                        }
                    }
                ]
            }
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes
        
        result = response.json()
        print("Gemini API response:\n", json.dumps(result, indent=2))

        # Extract the JSON text from the Gemini response
        gemini_text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Gemini may wrap the JSON in markdown code blocks, so we need to clean it
        if gemini_text.startswith("```json") and gemini_text.endswith("```"):
            gemini_text = gemini_text[7:-3].strip()

        # Parse the JSON and return it
        return json.loads(gemini_text)

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print("Raw Gemini text:", gemini_text)
        return None
    except Exception as e:
        print(f"An error occurred during Gemini analysis: {e}")
        return None

def save_document_to_storage(document_bytes):
    """
    Saves the original document to Google Cloud Storage.
    Returns the public URL of the saved document.
    """
    try:
        # Create a bucket if it doesn't exist
        bucket_name = f"{PROJECT_ID}-medical-documents"
        try:
            bucket = storage_client.get_bucket(bucket_name)
        except:
            bucket = storage_client.create_bucket(bucket_name)
            print(f"Bucket {bucket_name} created.")

        # Create a blob name with a unique identifier
        import uuid
        blob_name = f"documents/{uuid.uuid4()}.pdf" # Assuming PDF, adjust as needed

        # Upload the document
        blob = bucket.blob(blob_name)
        blob.upload_from_string(document_bytes, content_type="application/pdf")

        # Make the blob publicly readable (or set more specific permissions)
        blob.make_public()

        print(f"Document uploaded to {blob.public_url}")
        return blob.public_url

    except Exception as e:
        print(f"Error saving document to storage: {e}")
        return None

def save_data_to_firestore(patient_id, data, document_url):
    """
    Saves the analyzed data and document URL to a Firestore document.
    """
    try:
        # Create a new document in a 'medical_history' collection
        doc_ref = db.collection('medical_history').document(patient_id).collection('documents').document()
        
        # Add a timestamp and the document URL to the data
        data['uploadTimestamp'] = firestore.SERVER_TIMESTAMP
        data['originalDocumentURL'] = document_url
        
        doc_ref.set(data)
        print(f"Data saved to Firestore under document ID: {doc_ref.id}")

    except Exception as e:
        print(f"Error saving data to Firestore: {e}")
        return None

if __name__ == '__main__':
    app.run(port=5000, debug=True)
