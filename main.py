# main.py
#
# Python webhook logic for Dialogflow CX to handle document analysis.
# Uses: Google Cloud Vision, Gemini API, Cloud Storage, and Firestore
# to process a medical document and provide a structured summary.

import json
import os
import requests
from flask import Flask, request, jsonify
from google.cloud import vision_v1, storage, firestore

# -----------------------------------------------------------------------------
# Flask App Initialization
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Google Cloud Clients
# Make sure GOOGLE_APPLICATION_CREDENTIALS is set in your environment.
# -----------------------------------------------------------------------------
try:
    vision_client = vision_v1.ImageAnnotatorClient()
    storage_client = storage.Client()
    firestore_client = firestore.Client()
except Exception as e:
    print(f"Error initializing Google Cloud clients: {e}")
    # In production, handle this more gracefully.

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash-preview-05-20:generateContent"
)
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-gcs-bucket-name")

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def parse_dialogflow_session_id(session_path: str) -> str:
    """
    Parse Dialogflow session path to extract the session ID.
    Example: projects/PROJECT_ID/locations/LOCATION/agents/AGENT_ID/sessions/SESSION_ID
    """
    parts = session_path.split("/")
    return parts[8] if len(parts) >= 9 else None


def analyze_document_with_gemini(extracted_text: str):
    """
    Send the extracted text to Gemini API for structured analysis.
    Returns: (structured_data: dict | None, summary_text: str)
    """
    if not GEMINI_API_KEY:
        print("Gemini API key not found.")
        return None, "Analysis unavailable. Please try again later."

    # Prompt for structured data extraction
    prompt_for_structured_data = f"""
    Analyze the following medical lab report text. Extract the following information into a single JSON object.
    If any data is not present, use "N/A".
    - patientName (string)
    - testName (string)
    - testDate (string)
    - results (array of objects, each with 'resultName' and 'value')
    - interpretation (summary of the results and what they mean)

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
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=30
        )
        response.raise_for_status()
        gemini_response = response.json()
        structured_data_str = (
            gemini_response.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )
        structured_data = json.loads(structured_data_str)

        # Generate a friendly patient summary
        summary_prompt = f"""
        Given the following medical test results in JSON format, generate a concise,
        easy-to-read summary for the patient. Explain what the results mean,
        suggest next steps (like "discuss with your doctor"), and recommend follow-up actions.

        JSON data:
        ```
        {json.dumps(structured_data, indent=2)}
        ```
        """
        summary_payload = {
            "contents": [{"parts": [{"text": summary_prompt}]}],
            "generationConfig": {"responseMimeType": "text/plain"}
        }

        summary_response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            data=json.dumps(summary_payload),
            timeout=30
        )
        summary_response.raise_for_status()
        summary = (
            summary_response.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "Could not generate a summary.")
        )
        return structured_data, summary

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None, "There was a problem analyzing your document."


def save_to_firestore(app_id: str, user_id: str, doc_data: dict) -> bool:
    """
    Save the structured document data to Firestore.
    """
    try:
        doc_ref = (
            firestore_client.collection(
                f"artifacts/{app_id}/users/{user_id}/medical_reports"
            ).document()
        )
        doc_ref.set(doc_data)
        print(f"Data saved to Firestore: {doc_ref.path}")
        return True
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return False


def save_to_cloud_storage(app_id: str, user_id: str,
                          doc_name: str, doc_content: bytes) -> str | None:
    """
    Save the original document to a Google Cloud Storage bucket.
    Returns the gs:// URL if successful.
    """
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob_path = f"artifacts/{app_id}/users/{user_id}/medical_reports/{doc_name}"
        blob = bucket.blob(blob_path)
        blob.upload_from_string(doc_content)
        gcs_url = f"gs://{GCS_BUCKET_NAME}/{blob_path}"
        print(f"Document saved to Cloud Storage: {gcs_url}")
        return gcs_url
    except Exception as e:
        print(f"Error saving to Cloud Storage: {e}")
        return None

# -----------------------------------------------------------------------------
# Webhook Route
# -----------------------------------------------------------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Main webhook handler for Dialogflow CX requests.
    """
    req_body = request.get_json(silent=True)
    if not req_body:
        return jsonify({
            "fulfillmentResponse": {
                "messages": [{"text": {"text": ["Invalid request body."]}}]
            }
        })

    session_path = req_body.get("session", "")
    session_id = parse_dialogflow_session_id(session_path)
    if not session_id:
        return jsonify({
            "fulfillmentResponse": {
                "messages": [{"text": {"text": ["Could not identify session."]}}]
            }
        })

    # Use session ID as user identifier (simple approach)
    app_id = "default-app-id"
    user_id = session_id

    # Extract the document_url parameter
    document_url = None
    for param in req_body.get("pageInfo", {}).get("formInfo", {}).get("parameterInfo", []):
        if param.get("displayName") == "document_url" and "value" in param:
            document_url = param["value"]
            break

    # Fallback to session parameters
    if not document_url:
        document_url = req_body.get("sessionInfo", {}).get("parameters", {}).get("document_url")

    if not document_url:
        return jsonify({
            "fulfillmentResponse": {
                "messages": [{"text": {"text": ["No document URL provided."]}}]
            }
        })

    try:
        # Download the document
        doc_response = requests.get(document_url, stream=True, timeout=30)
        doc_response.raise_for_status()
        doc_content = doc_response.content
        doc_name = document_url.split("/")[-1]

        # Extract text using Google Cloud Vision
        image = vision_v1.Image(content=doc_content)
        response = vision_client.text_detection(image=image)
        full_text = (
            response.text_annotations[0].description
            if response.text_annotations else ""
        )

        if not full_text:
            return jsonify({
                "fulfillmentResponse": {
                    "messages": [{"text": {"text": [
                        "Could not extract text from the document. "
                        "Please ensure it is a clear image or PDF."
                    ]}}]
                }
            })

        # Analyze extracted text with Gemini API
        structured_data, summary_text = analyze_document_with_gemini(full_text)
        if not structured_data:
            return jsonify({
                "fulfillmentResponse": {
                    "messages": [{"text": {"text": [summary_text]}}]
                }
            })

        # Save to Cloud Storage and Firestore
        gcs_url = save_to_cloud_storage(app_id, user_id, doc_name, doc_content)
        structured_data["gcs_url"] = gcs_url
        save_to_firestore(app_id, user_id, structured_data)

        # Respond to Dialogflow
        return jsonify({
            "fulfillmentResponse": {
                "messages": [{
                    "text": {
                        "text": [
                            "Thank you! I have analyzed your document.",
                            f"Here is a summary of the results:\n\n{summary_text}"
                        ]
                    }
                }]
            }
        })

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        error_message = "I couldn't download the document. The link may have expired."
    except Exception as e:
        print(f"Unexpected error: {e}")
        error_message = "I encountered an error while processing your request. Please try again."

    return jsonify({
        "fulfillmentResponse": {
            "messages": [{"text": {"text": [error_message]}}]
        }
    })


# -----------------------------------------------------------------------------
# Local Development Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # For local testing
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
