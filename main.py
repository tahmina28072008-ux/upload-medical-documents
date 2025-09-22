# main.py
#
# This file contains the Python webhook logic for handling document analysis.
# It uses Dialogflow CX, Google Cloud Vision, Gemini API, Google Cloud Storage,
# and Firestore to process a medical document and provide a summary.

import json
import os
import requests
from flask import Flask, request, jsonify
from google.cloud import vision_v1, storage, firestore
from google.protobuf import json_format

# Initialize Flask app
app = Flask(__name__)

# Global clients - will be initialized in the webhook function
vision_client = None
storage_client = None
firestore_client = None

# --- API Keys and Configuration ---
# IMPORTANT: Use environment variables or a secrets manager for production.
# For this example, we'll use a placeholder.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Utility Functions ---

def initialize_clients():
    """
    Initializes Google Cloud clients from environment variables.
    """
    global vision_client, storage_client, firestore_client
    try:
        if not vision_client:
            vision_client = vision_v1.ImageAnnotatorClient()
        if not storage_client:
            storage_client = storage.Client()
        if not firestore_client:
            firestore_client = firestore.Client()
        print("Google Cloud clients initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Google Cloud clients: {e}")
        return False

def parse_dialogflow_session_id(session_path):
    """
    Parses a Dialogflow session path to extract the session ID.
    Example path: projects/PROJECT_ID/locations/LOCATION/agents/AGENT_ID/sessions/SESSION_ID
    """
    parts = session_path.split('/')
    if len(parts) >= 9:
        return parts[8]
    return None

def analyze_document_with_gemini(extracted_text):
    """
    Sends the extracted text to the Gemini API for analysis.
    The AI extracts structured data and generates a summary.
    """
    if not GEMINI_API_KEY:
        print("Gemini API key not found.")
        return None, "An error occurred during analysis. Please try again later."

    # Prompt for structured data extraction
    prompt_for_structured_data = f"""
    Analyze the following medical lab report text. Extract the following information into a single JSON object. If any data is not present, use "N/A".
    - patientName (string)
    - testName (string)
    - testDate (string)
    - results (array of objects, each with 'resultName' and 'value')
    - interpretation (a summary of the results and what they mean)

    Extracted text:
    ```
    {extracted_text}
    ```
    """

    payload = {
        "contents": [{"parts": [{"text": prompt_for_structured_data}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "patientName": {"type": "STRING"},
                    "testName": {"type": "STRING"},
                    "testDate": {"type": "STRING"},
                    "results": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "resultName": {"type": "STRING"},
                                "value": {"type": "STRING"}
                            }
                        }
                    },
                    "interpretation": {"type": "STRING"}
                }
            }
        }
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        gemini_response = response.json()
        structured_data_str = gemini_response.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
        structured_data = json.loads(structured_data_str)

        # Generate a friendly summary based on the structured data
        summary_prompt = f"""
        Given the following medical test results in JSON format, generate a concise, easy-to-read summary for the patient. Explain what the results mean, suggest next steps (like "discuss with your doctor"), and recommend follow-up actions.
        
        JSON data:
        ```
        {json.dumps(structured_data, indent=2)}
        ```
        """
        summary_payload = {
            "contents": [{"parts": [{"text": summary_prompt}]}],
            "generationConfig": {
                "responseMimeType": "text/plain"
            }
        }

        summary_response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(summary_payload)
        )
        summary_response.raise_for_status()
        summary = summary_response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Could not generate a summary.")

        return structured_data, summary
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None, "There was a problem analyzing your document."

def save_to_firestore(app_id, user_id, doc_data):
    """
    Saves the structured document data to Firestore.
    """
    try:
        doc_ref = firestore_client.collection(f"artifacts/{app_id}/users/{user_id}/medical_reports").document()
        doc_ref.set(doc_data)
        print(f"Data saved to Firestore: {doc_ref.path}")
        return True
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return False

def save_to_cloud_storage(app_id, user_id, doc_name, doc_content):
    """
    Saves the original document to a Google Cloud Storage bucket.
    """
    try:
        # Replace 'your-bucket-name' with your actual bucket name
        bucket = storage_client.bucket("your-gcs-bucket-name")
        blob_path = f"artifacts/{app_id}/users/{user_id}/medical_reports/{doc_name}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(doc_content)
        print(f"Document saved to Cloud Storage: gs://your-gcs-bucket-name/{blob_path}")
        return f"gs://your-gcs-bucket-name/{blob_path}"
    except Exception as e:
        print(f"Error saving to Cloud Storage: {e}")
        return None

# --- Main Webhook Route ---

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    Main webhook handler for Dialogflow CX requests.
    """
    if not initialize_clients():
        error_message = "I'm sorry, my internal systems are not configured correctly. Please contact the administrator."
        print(f"Failed to initialize Google Cloud clients. Returning error to Dialogflow.")
        return jsonify({
            "fulfillmentResponse": {
                "messages": [{"text": {"text": [error_message]}}]
            }
        })
    
    req_body = request.get_json(silent=True)
    # This line will print the full JSON request to your logs.
    print(f"Received request body: {json.dumps(req_body, indent=2)}")
    if not req_body:
        return jsonify({"fulfillmentResponse": {"messages": [{"text": {"text": ["Invalid request body."]}}]}})

    session_path = req_body.get('session', '')
    session_id = parse_dialogflow_session_id(session_path)
    if not session_id:
        return jsonify({"fulfillmentResponse": {"messages": [{"text": {"text": ["Could not identify session."]}}]}})

    # Use the session ID as a unique identifier for the user in this example
    # You would use a proper authentication system in a real-world app
    app_id = "default-app-id"  # Replace with a way to get your app ID, e.g., from environment
    user_id = session_id

    try:
        # 1. Get the document URL from the request parameters
        document_url = req_body.get('pageInfo', {}).get('formInfo', {}).get('parameterInfo', [{}])[0].get('value')
        
        # Fallback to the session parameters
        if not document_url:
            document_url = req_body.get('sessionInfo', {}).get('parameters', {}).get('document_url')

        if not document_url:
            return jsonify({"fulfillmentResponse": {"messages": [{"text": {"text": ["No document URL provided."]}}]}})

        print(f"Processing document from URL: {document_url}")

        # 2. Download the document
        doc_response = requests.get(document_url, stream=True)
        doc_response.raise_for_status()
        doc_content = doc_response.content
        doc_name = document_url.split('/')[-1]

        # 3. Extract text from the document using Google Cloud Vision
        image = vision_v1.Image(content=doc_content)
        response = vision_client.text_detection(image=image)
        full_text = response.text_annotations[0].description if response.text_annotations else ""

        if not full_text:
            return jsonify({"fulfillmentResponse": {"messages": [{"text": {"text": ["Could not extract any text from the document. Please ensure it is a clear image or PDF."]}}]}})

        # 4. Analyze text with Gemini API
        structured_data, summary_text = analyze_document_with_gemini(full_text)
        
        if not structured_data:
            return jsonify({"fulfillmentResponse": {"messages": [{"text": {"text": [summary_text]}}]}})

        # 5. Save original document to Cloud Storage
        gcs_url = save_to_cloud_storage(app_id, user_id, doc_name, doc_content)
        structured_data['gcs_url'] = gcs_url # Add the GCS URL to the data

        # 6. Save structured data to Firestore
        save_to_firestore(app_id, user_id, structured_data)

        # 7. Construct and return the Dialogflow CX response
        response_json = {
            "fulfillmentResponse": {
                "messages": [
                    {
                        "text": {
                            "text": [
                                f"Thank you! I have analyzed your document.",
                                f"Here is a summary of the results:\n\n{summary_text}"
                            ]
                        }
                    }
                ]
            }
        }
        return jsonify(response_json)

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        error_message = "I'm sorry, I couldn't download the document. The link may have expired."
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        error_message = "I encountered an error while processing your request. Please try again."
    
    return jsonify({
        "fulfillmentResponse": {
            "messages": [{"text": {"text": [error_message]}}]
        }
    })

if __name__ == '__main__':
    # For local development, use a different port and host
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
