import streamlit as st
from st_audiorec import st_audiorec
import requests
import openai
import psycopg2
import psycopg2.extras
import os
import time
import concurrent.futures
import uuid
import boto3
import io
import json
import logging

# ----------------------------------------------------------------------
# 1) Configure Logging
# ----------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# ----------------------------------------------------------------------
# 2) Read Secrets (Streamlit Cloud style)
# ----------------------------------------------------------------------
DEEPGRAM_API_KEY     = st.secrets["DEEPGRAM_API_KEY"]
OPENAI_API_KEY       = st.secrets["OPENAI_API_KEY"]
ASSEMBLYAI_API_KEY   = st.secrets["ASSEMBLYAI_API_KEY"]
DATABASE_URL         = st.secrets["DATABASE_URL"]
AWS_ACCESS_KEY_ID     = st.secrets["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
AWS_DEFAULT_REGION    = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
AWS_S3_BUCKET_NAME    = st.secrets["AWS_S3_BUCKET_NAME"]

# ----------------------------------------------------------------------
# 3) Configuration / Constants
# ----------------------------------------------------------------------
DEPARTMENTS = [
    "Cardiology",
    "Pulmonology",
    "Gastroenterology",
    "Neurology",
    "Rheumatology",
    "Dermatology",
    "Nephrology",
    "Hematology",
    "Infectious Diseases",
    "Psychiatry",
    "Pediatrics",
    "Orthopedics",
    "Ophthalmology",
    "Otolaryngology",
    "Gynecology",
    "Urology",
    "Oncology",
    "General Medicine",
    "Endocrinology",
]

# ----------------------------------------------------------------------
# 4) Upload audio bytes to S3
# ----------------------------------------------------------------------
def upload_audio_to_s3(audio_bytes: bytes) -> str:
    """Uploads audio bytes to S3 and returns a public or presigned URL."""
    if not AWS_S3_BUCKET_NAME:
        raise ValueError("Missing AWS_S3_BUCKET_NAME.")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_DEFAULT_REGION
    )

    unique_id = str(uuid.uuid4())
    object_key = f"audio_uploads/{unique_id}.wav"

    audio_stream = io.BytesIO(audio_bytes)
    s3.upload_fileobj(audio_stream, AWS_S3_BUCKET_NAME, object_key)

    audio_url = f"https://{AWS_S3_BUCKET_NAME}.s3.amazonaws.com/{object_key}"
    return audio_url

# ----------------------------------------------------------------------
# 5) Transcription Functions
# ----------------------------------------------------------------------
def transcribe_deepgram(audio_bytes: bytes) -> str:
    """Use Deepgram's medical model to transcribe the WAV bytes."""
    if not DEEPGRAM_API_KEY:
        logger.warning("Missing Deepgram API key.")
        return "Missing Deepgram API key."

    url = "https://api.deepgram.com/v1/listen?model=nova-2-medical"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    try:
        response = requests.post(url, headers=headers, data=audio_bytes)
        response.raise_for_status()
        result = response.json()
        transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
        return transcript.strip()
    except Exception as e:
        st.error(f"Deepgram transcription error: {e}")
        logger.error(f"Deepgram error: {e}", exc_info=True)
        return ""

def transcribe_whisper(audio_bytes: bytes) -> str:
    """
    Transcribe WAV bytes with the older approach (pinned openai==0.28.0).
    If on openai>=1.0.0, you'll need raw HTTP or revert to an older version.
    """
    if not OPENAI_API_KEY:
        logger.warning("Missing OpenAI API key.")
        return "Missing OpenAI API key."

    openai.api_key = OPENAI_API_KEY
    temp_file = "temp_whisper.wav"

    try:
        # 1) Write the bytes to a temp WAV file
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        # 2) Debug: log file size
        file_size = os.path.getsize(temp_file)
        logger.debug(f"Whisper WAV file size: {file_size} bytes")

        # 3) Transcribe (Note: old style, openai.Audio.transcribe)
        with open(temp_file, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)

        logger.info(f"Whisper response: {response}")
        text = response.get("text", "")
        return text.strip()

    except Exception as e:
        st.error(f"Whisper transcription error: {e}")
        logger.error(f"Whisper transcription error: {e}", exc_info=True)
        return ""

def transcribe_assemblyai(audio_bytes: bytes) -> str:
    """
    Transcribe WAV bytes with AssemblyAI (batch async).
    1) Upload audio
    2) Start transcription
    3) Poll for completion
    4) Return transcript
    """
    if not ASSEMBLYAI_API_KEY:
        logger.warning("Missing AssemblyAI API key.")
        return "Missing AssemblyAI API key."

    temp_file = "temp_assembly.wav"
    with open(temp_file, "wb") as f:
        f.write(audio_bytes)

    try:
        with open(temp_file, "rb") as f_data:
            data = f_data.read()
        upload_url = "https://api.assemblyai.com/v2/upload"
        headers = {"authorization": ASSEMBLYAI_API_KEY}
        upload_resp = requests.post(upload_url, headers=headers, data=data)
        upload_resp.raise_for_status()
        upload_result = upload_resp.json()
    except Exception as e:
        st.error(f"AssemblyAI upload failed: {e}")
        logger.error(f"AssemblyAI upload failed: {e}", exc_info=True)
        return ""

    if "upload_url" not in upload_result:
        return "AssemblyAI upload response invalid."

    audio_url = upload_result["upload_url"]
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"
    json_payload = {"audio_url": audio_url}

    try:
        start_resp = requests.post(
            transcript_endpoint,
            headers={"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"},
            json=json_payload
        )
        start_resp.raise_for_status()
        transcript_id = start_resp.json()["id"]
    except Exception as e:
        st.error(f"AssemblyAI transcription start error: {e}")
        logger.error(f"AssemblyAI start error: {e}", exc_info=True)
        return ""

    # Poll for completion
    polling_endpoint = f"{transcript_endpoint}/{transcript_id}"
    while True:
        try:
            poll_resp = requests.get(polling_endpoint, headers={"authorization": ASSEMBLYAI_API_KEY})
            poll_resp.raise_for_status()
            status_data = poll_resp.json()
            status = status_data["status"]

            if status == "completed":
                return status_data["text"].strip()
            elif status == "error":
                st.error("AssemblyAI transcription error: " + status_data.get("error", "Unknown"))
                logger.error(f"AssemblyAI transcription error response: {status_data}")
                return ""
            else:
                time.sleep(1)
        except Exception as e:
            st.error(f"AssemblyAI polling error: {e}")
            logger.error(f"AssemblyAI polling error: {e}", exc_info=True)
            return ""

def transcribe_all_in_parallel(audio_bytes: bytes):
    """
    Runs the three transcription functions in parallel threads.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_deepgram   = executor.submit(transcribe_deepgram, audio_bytes)
        future_whisper    = executor.submit(transcribe_whisper, audio_bytes)
        future_assemblyai = executor.submit(transcribe_assemblyai, audio_bytes)

        deepgram_text   = future_deepgram.result()
        whisper_text    = future_whisper.result()
        assemblyai_text = future_assemblyai.result()

    return deepgram_text, whisper_text, assemblyai_text

# ----------------------------------------------------------------------
# 6) Save to PostgreSQL
# ----------------------------------------------------------------------
def save_transcripts_to_postgres(
    dept: str,
    audio_url: str,
    deepgram_text: str,
    whisper_text: str,
    assemblyai_text: str,
    chosen_engine: str
):
    """Insert a record into the 'transcripts' table in PostgreSQL."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        insert_query = """
            INSERT INTO transcripts (
                dept,
                audio_url,
                deepgram_transcript,
                whisper_transcript,
                assemblyai_transcript,
                chosen_transcript
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(
            insert_query,
            (
                dept,
                audio_url,
                deepgram_text,
                whisper_text,
                assemblyai_text,
                chosen_engine
            )
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.error(f"Error saving transcript to PostgreSQL: {e}")
        logger.error(f"Postgres error: {e}", exc_info=True)

# ----------------------------------------------------------------------
# 7) Main Streamlit App
# ----------------------------------------------------------------------
def main():
    # Custom CSS
    st.markdown("""
        <style>
        body {
            background: #e6edf2;
        }
        .main .block-container {
            background-color: #fff;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            max-width: 800px;
            margin: 2rem auto;
        }
        h1 {
            margin-bottom: 1.2rem !important;
        }
        /* Increase department label font size significantly (by ~2) */
        .dept-label-custom {
            font-size: 2rem; 
            font-weight: 600;
            margin-top: 0.5rem;
            margin-bottom: 0.25rem;
        }
        /* Increase gap above the instruction box */
        .instruction-box {
            background-color: #f9f9f9;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 4px solid #2b78e4;
            border-radius: 4px;
            margin-top: 2rem; /* was 0.3rem */
        }
        /* Make tab labels bigger & bold (Deepgram, Whisper, AssemblyAI) */
        [data-testid="stTab-label"] {
            font-size: 1.5rem !important; /* Increase font size */
            font-weight: 700 !important;  /* Make text bold */
        }
        
        /* Make the "Save Transcript" button more attractive */
        .stButton > button {
            background-color: #2b78e4 !important;
            color: #fff !important;
            font-weight: 600 !important;
            border-radius: 6px !important;
            border: none !important;
            padding: 0.6rem 1.2rem !important;
            transition: background-color 0.2s ease, transform 0.2s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        }
        .stButton > button:hover {
            background-color: #1f5bb5 !important;
            transform: translateY(-2px);
        }
        .stRadio label {
            font-weight: 500; 
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("Voice to Text Testing")

    # Check missing credentials
    missing_keys = []
    if not DEEPGRAM_API_KEY:
        missing_keys.append("Deepgram")
    if not OPENAI_API_KEY:
        missing_keys.append("OpenAI")
    if not ASSEMBLYAI_API_KEY:
        missing_keys.append("AssemblyAI")

    if missing_keys:
        st.error(f"Missing keys for: {', '.join(missing_keys)}")
        return

    if not DATABASE_URL:
        st.error("Missing DATABASE_URL.")
        return

    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and AWS_S3_BUCKET_NAME):
        st.error("Missing AWS S3 credentials or bucket name.")
        return

    # Department
    st.markdown("<p class='dept-label-custom'>Department</p>", unsafe_allow_html=True)
    selected_department = st.selectbox("", DEPARTMENTS)

    # Instructions
    st.markdown("""
    <div class='instruction-box'>
      <p><strong>Instructions:</strong><br>
      1) Click on the 'Start Recording' button, and then click 'Stop' after recording.<br>
      2) Wait for auto-transcription.<br>
      3) Choose the best transcript and click "Save".
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Audio Recorder
    audio_data = st_audiorec()

    if audio_data:
        st.write(f"Received {len(audio_data)} bytes of audio data.")
    else:
        st.info("No audio recorded. Click the microphone to record.")

    # If audio is available, do S3 upload + transcribe
    if audio_data:
        with st.spinner("Uploading to S3..."):
            try:
                audio_url = upload_audio_to_s3(audio_data)
            except Exception as e:
                st.error(f"S3 upload error: {e}")
                logger.error(f"S3 upload error: {e}", exc_info=True)
                return

        with st.spinner("Transcribing..."):
            deepgram_text, whisper_text, assemblyai_text = transcribe_all_in_parallel(audio_data)

        st.success("Transcription completed!")

        # Display transcripts
        st.subheader("Transcribed Results")
        tabs = st.tabs(["Deepgram", "Whisper", "AssemblyAI"])
        with tabs[0]:
            st.write(deepgram_text if deepgram_text else "_No transcript_")

        with tabs[1]:
            st.write(whisper_text if whisper_text else "_No transcript_")

        with tabs[2]:
            st.write(assemblyai_text if assemblyai_text else "_No transcript_")

        # Choose best transcript
        st.subheader("Choose the most accurate transcript")
        choice = st.radio(
            label="Select transcript",
            options=["Deepgram", "Whisper", "AssemblyAI"],
            index=0,
            label_visibility="collapsed"
        )

        # Save to database
        if st.button("Save"):
            if not (deepgram_text.strip() or whisper_text.strip() or assemblyai_text.strip()):
                st.warning("All transcripts empty. Cannot save.")
            else:
                save_transcripts_to_postgres(
                    dept=selected_department,
                    audio_url=audio_url,
                    deepgram_text=deepgram_text,
                    whisper_text=whisper_text,
                    assemblyai_text=assemblyai_text,
                    chosen_engine=choice
                )
                st.success("Record saved!")

# ----------------------------------------------------------------------
# 8) Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
