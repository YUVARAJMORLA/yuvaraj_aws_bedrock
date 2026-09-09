import os
import io
import wave
import json
import uuid
import base64
import asyncio
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import boto3
import pyaudio
import pypdf
import pandas as pd
import streamlit as st

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    import docx
except ImportError:
    docx = None

try:
    import pptx
except ImportError:
    pptx = None

from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart
)
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver


# ============================================================
# AWS CREDENTIALS SYNCHRONIZATION
# ============================================================
def ensure_aws_credentials():
    try:
        session = boto3.Session()
        creds = session.get_credentials()
        if creds:
            frozen = creds.get_frozen_credentials()
            if frozen.access_key and "AWS_ACCESS_KEY_ID" not in os.environ:
                os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
            if frozen.secret_key and "AWS_SECRET_ACCESS_KEY" not in os.environ:
                os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
            if frozen.token and "AWS_SESSION_TOKEN" not in os.environ:
                os.environ["AWS_SESSION_TOKEN"] = frozen.token
        if "AWS_DEFAULT_REGION" not in os.environ:
            os.environ["AWS_DEFAULT_REGION"] = AWS_REGION
    except Exception as e:
        print(f"Credentials setup warning: {e}")

AWS_REGION = "us-east-1"
ensure_aws_credentials()


# ============================================================
# CONFIGURATION & MODELS
# ============================================================
EMBEDDING_MODEL = "amazon.titan-embed-text-v1"

# Model 1: Llama 4 (Text-to-Text Chatbot RAG)
LLAMA_MODEL_ID = os.environ.get(
    "LLAMA_MODEL_ID",
    (
        "arn:aws:bedrock:us-east-1:776817040428:"
        "inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0"
    )
)

# Model 2: Amazon Nova 2 Sonic (Live Speech-to-Speech Voice)
NOVA_SONIC_MODEL_ID = "amazon.nova-2-sonic-v1:0"

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16

CHUNK_SIZE = 1024
OUTPUT_CHUNK_SIZE = 2048

OUT_OF_CONTEXT = (
    "This question is out of context. "
    "I can only answer questions based on the uploaded documents."
)


# ============================================================
# AWS CLIENTS
# ============================================================
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

bedrock_embeddings = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL,
    client=bedrock
)


# ============================================================
# STREAMLIT PAGE CONFIG & SESSION STATE
# ============================================================
st.set_page_config(
    page_title="Document AI Assistant — Llama & Nova",
    page_icon="📚",
    layout="wide"
)

defaults = {
    "vectorstore": None,
    "documents_processed": False,
    "uploaded_files": [],
    "document_summary": "",
    "llama_messages": [],
    "nova_thread": None,
    "nova_running": False,
    "nova_status": "Not connected",
    "nova_messages": [],
    "nova_error": None,
    "last_spoken_query": "",
    "last_retrieved_sources": [],
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================
# MULTI-FORMAT DOCUMENT INGESTION (PDF, Word, TXT, MD, CSV)
# High-performance in-memory parsing for 100MB+ / 250+ page files
# ============================================================
def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        return []

    documents = []

    # 1. PDF Documents (PyMuPDF high-speed engine + pypdf fallback)
    if ext == ".pdf":
        pymupdf_success = False
        if pymupdf is not None:
            try:
                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                for i, page in enumerate(doc):
                    try:
                        text = page.get_text().strip()
                        if text:
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1}
                            ))
                    except Exception:
                        continue
                doc.close()
                if documents:
                    pymupdf_success = True
            except Exception as fitz_err:
                print(f"PyMuPDF notice for {filename}: {fitz_err}. Trying pypdf fallback...")

        if not pymupdf_success:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes), strict=False)
                for i, page in enumerate(reader.pages):
                    try:
                        text = (page.extract_text() or "").strip()
                        if text:
                            documents.append(Document(
                                page_content=text,
                                metadata={"source": filename, "page": i + 1}
                            ))
                    except Exception:
                        continue
            except Exception as pdf_fallback_err:
                print(f"pypdf fallback error on {filename}: {pdf_fallback_err}")

    # 2. Word Documents (.docx, .doc)
    elif ext in [".docx", ".doc"]:
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        paragraphs.append(row_text)
            full_text = "\n\n".join(paragraphs).strip()
            if full_text:
                documents.append(Document(
                    page_content=full_text,
                    metadata={"source": filename, "page": 1}
                ))
        except Exception as docx_err:
            print(f"Word docx parsing notice for {filename}: {docx_err}. Attempting raw stream extraction...")
            try:
                raw_text = file_bytes.decode("utf-8", errors="ignore")
                printable = "".join(ch for ch in raw_text if ch.isprintable() or ch in "\n\r\t")
                clean_lines = [line.strip() for line in printable.splitlines() if len(line.strip()) > 3]
                if clean_lines:
                    documents.append(Document(
                        page_content="\n".join(clean_lines),
                        metadata={"source": filename, "page": 1}
                    ))
            except Exception:
                pass

    # 3. PowerPoint Presentations (.pptx, .ppt)
    elif ext in [".pptx", ".ppt"]:
        try:
            prs = pptx.Presentation(io.BytesIO(file_bytes))
            for i, slide in enumerate(prs.slides):
                slide_texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            line = para.text.strip()
                            if line:
                                slide_texts.append(line)
                    elif shape.has_table:
                        for row in shape.table.rows:
                            row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                            if row_text:
                                slide_texts.append(row_text)
                if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                    notes = slide.notes_slide.notes_text_frame.text.strip()
                    if notes:
                        slide_texts.append(f"Notes: {notes}")

                if slide_texts:
                    documents.append(Document(
                        page_content="\n".join(slide_texts),
                        metadata={"source": filename, "page": i + 1}
                    ))
        except Exception as ppt_err:
            print(f"PowerPoint parsing notice for {filename}: {ppt_err}")

    # 4. Excel Spreadsheets (.xlsx, .xls)
    elif ext in [".xlsx", ".xls"]:
        try:
            excel_data = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue
                sheet_lines = [f"Sheet: {sheet_name}", "Columns: " + ", ".join([str(c) for c in df.columns])]
                for _, row in df.iterrows():
                    row_strs = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
                    if row_strs:
                        sheet_lines.append(" | ".join(row_strs))
                sheet_text = "\n".join(sheet_lines).strip()
                if sheet_text:
                    documents.append(Document(
                        page_content=sheet_text,
                        metadata={"source": filename, "page": str(sheet_name)}
                    ))
        except Exception as xl_err:
            print(f"Excel parsing notice for {filename}: {xl_err}")

    # 5. CSV and TSV
    elif ext in [".csv", ".tsv"]:
        try:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, on_bad_lines="skip")
            csv_lines = ["Columns: " + ", ".join([str(c) for c in df.columns])]
            for _, row in df.iterrows():
                row_strs = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()]
                if row_strs:
                    csv_lines.append(" | ".join(row_strs))
            csv_text = "\n".join(csv_lines).strip()
            if csv_text:
                documents.append(Document(
                    page_content=csv_text,
                    metadata={"source": filename, "page": 1}
                ))
        except Exception:
            for enc in ["utf-8", "utf-8-sig", "cp1252", "latin-1"]:
                try:
                    text = file_bytes.decode(enc).strip()
                    if text:
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": filename, "page": 1}
                        ))
                    break
                except UnicodeDecodeError:
                    continue

    # 6. Text, Markdown, JSON, Code, Log files
    else:
        text = ""
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1", "gbk", "shift_jis"]
        for enc in encodings:
            try:
                text = file_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue

        if not text:
            text = file_bytes.decode("utf-8", errors="ignore")

        text = text.strip()
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": filename, "page": 1}
            ))

    return documents


# ============================================================
# CLEAN DOCUMENT KNOWLEDGE SUMMARY (NOVA SONIC RAG)
# Evenly samples large documents (250+ pages) without filter errors
# ============================================================
def generate_document_summary(all_documents):
    if not all_documents:
        return "No text available."

    total_docs = len(all_documents)
    sampled_texts = []
    sample_indices = set()

    # Beginning (pages 1 to 3)
    for i in range(min(3, total_docs)):
        sample_indices.add(i)

    # Middle intervals (25%, 50%, 75%)
    if total_docs > 6:
        for pct in [0.25, 0.50, 0.75]:
            idx = int(total_docs * pct)
            sample_indices.add(min(idx, total_docs - 1))

    # End
    if total_docs > 10:
        for i in range(max(0, total_docs - 3), total_docs):
            sample_indices.add(i)

    for idx in sorted(list(sample_indices)):
        text = all_documents[idx].page_content.strip()
        if text:
            sampled_texts.append(text[:1500])

    raw_sample = "\n\n".join(sampled_texts)[:7500]
    if not raw_sample.strip():
        return "No readable content found."

    prompt = f"""You are an expert document summarizer.
Create a comprehensive, factual, clean knowledge summary of the uploaded document(s) below for an AI voice assistant.
Include all key facts, company/document overview, major business segments, products, services, financial highlights, and important topics.
Keep the output strictly factual, clear, and professional. Do NOT include raw boilerplate legal disclaimers.

DOCUMENT TEXT:
{raw_sample}
"""

    try:
        response = bedrock.converse(
            modelId=LLAMA_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.2}
        )
        content_list = response.get("output", {}).get("message", {}).get("content", [])
        if content_list and "text" in content_list[0]:
            return content_list[0]["text"].strip()
        return raw_sample[:3000]
    except Exception as e:
        print(f"Summary generation notice: {e}")
        return raw_sample[:3000]


# ============================================================
# HIGH-THROUGHPUT CONCURRENT FAISS INDEXING FOR LARGE FILES
# Eliminates Bedrock Throttling and IndexError: list index out of range
# ============================================================
def create_faiss_index_concurrent(docs, progress_bar=None, status_text=None):
    if not docs:
        raise ValueError("No text chunks provided for vector indexing.")

    # 1. Clean and filter chunks: ensure real text, remove whitespace
    clean_items = []
    for d in docs:
        cleaned = d.page_content.strip() if d.page_content else ""
        if len(cleaned) >= 15:
            clean_items.append((cleaned, d.metadata))

    if not clean_items:
        raise ValueError("All extracted chunks were empty or too short for embedding.")

    total_chunks = len(clean_items)
    batch_size = 35
    all_embedded = []

    def embed_single_chunk(text, meta):
        # Retry loop for Bedrock rate limits with backoff
        for attempt in range(4):
            try:
                body = json.dumps({"inputText": text})
                resp = bedrock.invoke_model(
                    modelId=EMBEDDING_MODEL,
                    contentType="application/json",
                    accept="application/json",
                    body=body
                )
                vec = json.loads(resp["body"].read().decode("utf-8"))["embedding"]
                return (text, vec, meta)
            except Exception as e:
                if attempt == 3:
                    print(f"Skipping chunk due to persistent Bedrock error: {e}")
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    # Embed in concurrent batches
    for i in range(0, total_chunks, batch_size):
        batch = clean_items[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(embed_single_chunk, text, meta) for text, meta in batch]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    all_embedded.append(res)

        done = min(i + batch_size, total_chunks)
        if progress_bar:
            progress_bar.progress(done / total_chunks)
        if status_text:
            status_text.text(f"Generated embeddings for {done}/{total_chunks} chunks ({int(done/total_chunks*100)}%)...")

    if not all_embedded:
        raise RuntimeError("Vector embedding failed for all document chunks. Please check AWS Bedrock permissions and rate limits.")

    texts = [item[0] for item in all_embedded]
    vectors = [item[1] for item in all_embedded]
    metas = [item[2] for item in all_embedded]

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=bedrock_embeddings,
        metadatas=metas
    )
    return vectorstore


def process_documents(uploaded_files, progress_bar=None, status_text=None):
    all_documents = []

    # 1. Extract text from all files
    for idx, uploaded_file in enumerate(uploaded_files):
        if status_text:
            status_text.text(f"Reading file {idx+1}/{len(uploaded_files)}: {uploaded_file.name}...")
        file_docs = extract_text_from_file(uploaded_file)
        all_documents.extend(file_docs)

    if not all_documents or not any(d.page_content and len(d.page_content.strip()) >= 10 for d in all_documents):
        raise ValueError(
            "No readable text could be extracted from the uploaded document(s). "
            "If uploading scanned PDFs or images, please ensure the file has a searchable text layer."
        )

    # 2. Split into clean chunks
    if status_text:
        status_text.text(f"Splitting {len(all_documents)} document sections into semantic chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    docs = splitter.split_documents(all_documents)
    # Filter out empty or whitespace-only chunks
    docs = [d for d in docs if d.page_content and len(d.page_content.strip()) >= 20]

    if not docs:
        raise ValueError("Document yielded no valid text chunks after splitting. Please ensure documents contain readable text.")

    # 3. High-throughput concurrent FAISS embedding with live progress
    if status_text:
        status_text.text(f"Generating Titan embeddings for {len(docs)} chunks...")

    vectorstore = create_faiss_index_concurrent(docs, progress_bar=progress_bar, status_text=status_text)
    if vectorstore is None:
        raise RuntimeError("Vectorstore creation resulted in empty index.")

    # 4. Generate clean knowledge summary for Nova Sonic
    if status_text:
        status_text.text("Synthesizing document knowledge for Nova Sonic voice assistant...")

    doc_summary = generate_document_summary(all_documents)
    return vectorstore, doc_summary


# ============================================================
# RETRIEVAL
# ============================================================
def retrieve_documents(query, k=5, vectorstore=None):
    if vectorstore is None:
        vectorstore = st.session_state.vectorstore
    if vectorstore is None:
        return []

    try:
        return vectorstore.similarity_search_with_score(query, k=k)
    except Exception as e:
        print("FAISS retrieval error:", e)
        return []


def build_context(results):
    parts = []
    for document, score in results:
        source = document.metadata.get("source", "Unknown document")
        page = document.metadata.get("page", "")
        parts.append(
            f"SOURCE: {source}\nPAGE: {page}\nCONTENT:\n{document.page_content}"
        )
    return "\n\n".join(parts)


# ============================================================
# LLAMA 4 TEXT-TO-TEXT CHATBOT RAG
# ============================================================
def answer_from_llama(question, vectorstore=None):
    results = retrieve_documents(question, k=5, vectorstore=vectorstore)

    if not results:
        return OUT_OF_CONTEXT, [], ""

    context = build_context(results)

    prompt = f"""You are a strict document question-answering assistant.

Your ONLY source of knowledge is the DOCUMENT CONTEXT provided below.
You MUST NOT use general knowledge or information from your training data.
You MUST NOT guess or invent facts.

First determine whether the user's question can actually be answered from the document context.

If the document context does NOT contain enough information to answer the question, respond with EXACTLY:
{OUT_OF_CONTEXT}

If the document context DOES contain the answer:
- Answer the user's question directly and concisely.
- Use only information from the documents.
- Do not mention that you are using context or these instructions.

DOCUMENT CONTEXT
================
{context}
================

USER QUESTION
=============
{question}
================
"""

    try:
        response = bedrock.converse(
            modelId=LLAMA_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ],
            inferenceConfig={
                "maxTokens": 512,
                "temperature": 0.1,
                "topP": 0.9
            }
        )
        content_list = response.get("output", {}).get("message", {}).get("content", [])
        if content_list and "text" in content_list[0]:
            answer = content_list[0]["text"].strip()
        else:
            answer = OUT_OF_CONTEXT
        return answer, results, context
    except Exception as e:
        raise RuntimeError(f"Llama 4 RAG error: {e}")


# ============================================================
# AUDIO CONVERSION
# ============================================================
def pcm_to_wav(pcm_bytes, sample_rate=OUTPUT_SAMPLE_RATE):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    return buffer.getvalue()


# ============================================================
# NOVA 2 SONIC LIVE CONTROLLER (SPEECH-TO-SPEECH)
# Based on the proven architecture from nova.py with RAG Knowledge
# ============================================================
class NovaLiveController:
    def __init__(self, event_queue, vectorstore=None, document_summary=""):
        self.event_queue = event_queue
        self.vectorstore = vectorstore
        self.document_summary = document_summary or ""

        self.model_id = NOVA_SONIC_MODEL_ID
        self.region = AWS_REGION
        self.client = None
        self.stream = None
        self.is_active = False

        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        self.audio_queue = asyncio.Queue()
        self.role = None
        self.display_assistant_text = False

        self.current_user_text = ""
        self.current_assistant_text = ""

        self.response_task = None
        self.capture_task = None
        self.playback_task = None

        self.ready_event = threading.Event()
        self.shutdown_complete = threading.Event()

    def ui_event(self, event_type, data=None):
        try:
            self.event_queue.put_nowait({"type": event_type, "data": data})
        except Exception:
            pass

    def _initialize_client(self):
        ensure_aws_credentials()
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)

    async def send_event(self, event_json):
        if not self.stream:
            return
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(
                bytes_=event_json.encode('utf-8')
            )
        )
        await self.stream.input_stream.send(event)

    async def start_session(self):
        self._initialize_client()
        self.ui_event("status", "Connecting to Amazon Nova Sonic...")

        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(
                model_id=self.model_id
            )
        )

        self.is_active = True

        # 1. sessionStart
        session_start = '''
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.6
              }
            }
          }
        }
        '''
        await self.send_event(session_start)

        # 2. promptStart
        prompt_start = f'''
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{self.prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 24000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "matthew",
                "encoding": "base64",
                "audioType": "SPEECH"
              }}
            }}
          }}
        }}
        '''
        await self.send_event(prompt_start)

        # 3. Clean, Grounded System Prompt
        text_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "type": "TEXT",
                    "interactive": false,
                    "role": "SYSTEM",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(text_content_start)

        system_prompt = (
            "You are a warm, professional, and helpful document AI voice assistant. "
            "You help the user explore and understand their uploaded documents. "
            f"Here is the knowledge from the uploaded documents:\n\n{self.document_summary}\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's questions clearly, accurately, and naturally based on the document knowledge above.\n"
            "2. Keep your answers concise, typically within 2 to 4 spoken sentences.\n"
            "3. If the user asks about something completely outside the uploaded documents, politely let them know that you can only answer questions based on their uploaded documents.\n"
            "4. Speak directly, naturally, and warmly without sounding like an essay."
        )

        clean_prompt_json = json.dumps(system_prompt)

        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "content": {clean_prompt_json}
                }}
            }}
        }}
        '''
        await self.send_event(text_input)

        text_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(text_content_end)

        # 4. Background tasks
        self.response_task = asyncio.create_task(self._process_responses())
        self.playback_task = asyncio.create_task(self.play_audio())
        self.capture_task = asyncio.create_task(self.capture_audio())

        self.ui_event("status", "🎤 Listening... Speak into your microphone.")
        self.ready_event.set()

    async def start_audio_input(self):
        audio_content_start = f'''
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_start)

    async def send_audio_chunk(self, audio_bytes):
        if not self.is_active:
            return
        blob = base64.b64encode(audio_bytes)
        audio_event = f'''
        {{
            "event": {{
                "audioInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "content": "{blob.decode('utf-8')}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_event)

    async def end_audio_input(self):
        audio_content_end = f'''
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}"
                }}
            }}
        }}
        '''
        await self.send_event(audio_content_end)

    async def capture_audio(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=INPUT_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
        except Exception as e:
            self.ui_event("error", f"Microphone access error: {e}")
            return

        await self.start_audio_input()

        try:
            while self.is_active:
                audio_data = await asyncio.to_thread(
                    stream.read,
                    CHUNK_SIZE,
                    exception_on_overflow=False
                )
                await self.send_audio_chunk(audio_data)
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.is_active:
                self.ui_event("error", f"Audio capture error: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass
            try:
                await self.end_audio_input()
            except Exception:
                pass

    async def play_audio(self):
        p = pyaudio.PyAudio()
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=OUTPUT_SAMPLE_RATE,
                output=True,
                frames_per_buffer=2048
            )
        except Exception as e:
            self.ui_event("error", f"Audio output error: {e}")
            return

        try:
            while self.is_active:
                audio_data = await self.audio_queue.get()
                for i in range(0, len(audio_data), OUTPUT_CHUNK_SIZE):
                    if not self.is_active:
                        break
                    end = min(i + OUTPUT_CHUNK_SIZE, len(audio_data))
                    chunk = audio_data[i:end]
                    await asyncio.to_thread(stream.write, chunk)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.is_active:
                self.ui_event("error", f"Audio playback error: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass

    async def _process_responses(self):
        try:
            while self.is_active:
                output = await self.stream.await_output()
                result = await output[1].receive()

                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)

                    if 'event' in json_data:
                        ev = json_data['event']

                        # 1. contentStart
                        if 'contentStart' in ev:
                            content_start = ev['contentStart']
                            self.role = content_start.get('role')

                            if 'additionalModelFields' in content_start:
                                additional_fields = json.loads(
                                    content_start['additionalModelFields']
                                )
                                if additional_fields.get('generationStage') == 'SPECULATIVE':
                                    self.display_assistant_text = True
                                else:
                                    self.display_assistant_text = False

                        # 2. textOutput (Display user speech & Nova speech as text)
                        elif 'textOutput' in ev:
                            text = ev['textOutput'].get('content', '')

                            if self.role == "ASSISTANT" and self.display_assistant_text:
                                self.current_assistant_text += text
                                self.ui_event("status", "🔊 Nova Sonic is speaking...")

                            elif self.role == "USER":
                                self.current_user_text += text

                        # 3. audioOutput (Nova voice speech to speakers)
                        elif 'audioOutput' in ev:
                            audio_content = ev['audioOutput'].get('content', '')
                            if audio_content:
                                audio_bytes = base64.b64decode(audio_content)
                                await self.audio_queue.put(audio_bytes)

                        # 4. contentEnd (Finalize turn)
                        elif 'contentEnd' in ev:
                            # User turn finished
                            if self.current_user_text.strip():
                                final_user_text = self.current_user_text.strip()
                                self.current_user_text = ""
                                self.ui_event("user_final", final_user_text)

                                # Search documents matching what user asked
                                if self.vectorstore:
                                    sources = retrieve_documents(final_user_text, k=5, vectorstore=self.vectorstore)
                                    source_details = [
                                        {
                                            "source": doc.metadata.get("source", "Unknown"),
                                            "page": doc.metadata.get("page", ""),
                                            "score": float(score),
                                            "content": doc.page_content
                                        }
                                        for doc, score in sources
                                    ]
                                    self.ui_event("retrieved_sources", {
                                        "question": final_user_text,
                                        "sources": source_details
                                    })

                            # Assistant turn finished
                            if self.current_assistant_text.strip():
                                final_assistant_text = self.current_assistant_text.strip()
                                self.current_assistant_text = ""
                                self.ui_event("assistant_final", final_assistant_text)
                                self.ui_event("status", "🎤 Listening... Ask your next question.")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self.is_active:
                self.ui_event("error", f"Nova response error: {e}")

    async def end_session(self):
        if not self.is_active or self.shutdown_complete.is_set():
            return

        self.ui_event("status", "Ending Nova Sonic session...")
        self.is_active = False

        prompt_end = f'''
        {{
            "event": {{
                "promptEnd": {{
                    "promptName": "{self.prompt_name}"
                }}
            }}
        }}
        '''
        try:
            await self.send_event(prompt_end)
        except Exception:
            pass

        session_end = '''
        {
            "event": {
                "sessionEnd": {}
            }
        }
        '''
        try:
            await self.send_event(session_end)
        except Exception:
            pass

        try:
            if self.stream:
                await self.stream.input_stream.close()
        except Exception:
            pass

        tasks = [self.playback_task, self.capture_task, self.response_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()

        for task in tasks:
            if task:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        self.playback_task = None
        self.capture_task = None
        self.response_task = None
        self.shutdown_complete.set()
        self.ui_event("status", "Session ended")


# ============================================================
# NOVA THREAD CONTROLLER
# ============================================================
class NovaThread:
    def __init__(self, vectorstore, document_summary):
        self.events = queue.Queue()
        self.controller = NovaLiveController(
            event_queue=self.events,
            vectorstore=vectorstore,
            document_summary=document_summary
        )
        self.loop = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.started = False

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.controller.start_session())
            self.loop.run_forever()
        except Exception as e:
            self.events.put({"type": "error", "data": str(e)})
        finally:
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            try:
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception:
                pass

    def start(self):
        if self.started:
            return
        self.started = True
        self.thread.start()

        if not self.controller.ready_event.wait(timeout=15):
            raise RuntimeError("Nova Sonic did not start within 15 seconds. Check microphone & AWS connection.")

    def stop(self):
        if not self.loop:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.controller.end_session(),
                self.loop
            )
            future.result(timeout=10)
        except Exception as e:
            print(f"Nova shutdown notice: {e}")

        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass

        try:
            self.thread.join(timeout=5)
        except Exception:
            pass


def start_nova():
    if st.session_state.nova_running:
        return

    st.session_state.nova_messages = []
    st.session_state.nova_error = None
    st.session_state.nova_status = "Starting Amazon Nova Sonic..."

    vectorstore = st.session_state.vectorstore
    doc_summary = st.session_state.document_summary

    nova = NovaThread(vectorstore, doc_summary)
    st.session_state.nova_thread = nova
    st.session_state.nova_running = True

    try:
        nova.start()
    except Exception as e:
        st.session_state.nova_running = False
        st.session_state.nova_thread = None
        st.session_state.nova_error = str(e)
        st.session_state.nova_status = "❌ Failed to start"


def stop_nova():
    nova = st.session_state.nova_thread
    if nova:
        nova.stop()
    st.session_state.nova_running = False
    st.session_state.nova_status = "Session ended"
    st.session_state.nova_thread = None


def process_nova_events():
    nova = st.session_state.nova_thread
    if nova is None:
        return

    while True:
        try:
            event = nova.events.get_nowait()
        except queue.Empty:
            break

        event_type = event["type"]
        data = event["data"]

        if event_type == "status":
            st.session_state.nova_status = data
        elif event_type == "error":
            st.session_state.nova_error = data
            st.session_state.nova_status = "❌ Error"
        elif event_type == "user_final":
            question = data.strip()
            if question:
                st.session_state.nova_messages.append({
                    "role": "user",
                    "content": question
                })
                st.session_state.last_spoken_query = question
        elif event_type == "assistant_final":
            answer = data.strip()
            if answer:
                st.session_state.nova_messages.append({
                    "role": "assistant",
                    "content": answer
                })
        elif event_type == "retrieved_sources":
            st.session_state.last_retrieved_sources = data.get("sources", [])


# ============================================================
# LIVE NOVA UI FRAGMENT
# ============================================================
@st.fragment(run_every=0.5)
def live_nova_ui():
    process_nova_events()

    st.info(f"**Live Status:** {st.session_state.nova_status}")

    if st.session_state.nova_error:
        st.error(st.session_state.nova_error)

    st.markdown("### 💬 Live Voice Conversation")
    if not st.session_state.nova_messages:
        st.write("Click **Start Voice Chat** and begin speaking into your microphone.")
    else:
        for message in st.session_state.nova_messages:
            role = message["role"]
            with st.chat_message(role):
                if role == "assistant":
                    st.write(f"**Nova Sonic:** {message['content']}")
                else:
                    st.write(f"**You:** {message['content']}")

    # Document Sources matched to what was spoken
    if st.session_state.last_retrieved_sources:
        st.markdown("### 📄 Retrieved Document Sources for Spoken Question")
        st.caption(f"Document sources matching: *\"{st.session_state.last_spoken_query}\"*")
        for src in st.session_state.last_retrieved_sources:
            st.markdown(
                f"• **{src['source']}** (Page {src['page']}) — *FAISS Distance: {src['score']:.4f}*"
            )
        with st.expander("🔍 View Retrieved Document Chunks"):
            for i, src in enumerate(st.session_state.last_retrieved_sources):
                st.markdown(f"**Chunk {i+1} — {src['source']} (Page {src['page']})**")
                st.code(src.get("content", ""))

    if st.session_state.nova_running:
        st.success("🎤 **Microphone active** — Speak naturally. Nova Sonic will listen, answer with voice, and display text.")


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    st.title("📚 Document AI Assistant — Llama & Nova")
    st.write(
        "Dual-model Document AI: **Llama 4** provides an interactive conversational **Text-to-Text RAG Chatbot**, "
        "and **Amazon Nova 2 Sonic** provides real-time conversational **Voice-to-Voice AI with live text transcriptions**."
    )

    # --------------------------------------------------------
    # SIDEBAR: DOCUMENT INGESTION
    # --------------------------------------------------------
    with st.sidebar:
        st.header("📄 Document Ingestion")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, Word, PPTX, Excel, TXT, MD, CSV, etc.)",
            type=[
                "pdf", "docx", "doc", "pptx", "ppt",
                "xlsx", "xls", "csv", "tsv",
                "txt", "md", "json", "log", "xml", "html", "rst", "py", "sql", "yaml", "yml"
            ],
            accept_multiple_files=True
        )

        if uploaded_files:
            total_size_mb = sum(len(f.getvalue()) for f in uploaded_files) / (1024 * 1024)
            st.write(f"**{len(uploaded_files)}** document(s) selected ({total_size_mb:.1f} MB).")

        if st.button("⚙️ Process Documents", use_container_width=True):
            if not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                try:
                    vectorstore, doc_summary = process_documents(
                        uploaded_files,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.document_summary = doc_summary
                    st.session_state.documents_processed = True
                    st.session_state.uploaded_files = [f.name for f in uploaded_files]

                    progress_bar.progress(1.0)
                    status_text.text("✅ Indexing complete!")
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    status_text.empty()

        if st.session_state.documents_processed:
            st.success("🟢 Document Index Ready")
            for filename in st.session_state.uploaded_files:
                st.write(f"• {filename}")

            with st.expander("📋 View Document Summary"):
                st.write(st.session_state.document_summary)

    if not st.session_state.documents_processed:
        st.info("👈 Please upload documents in the sidebar and click **Process Documents** to get started.")
        return

    # --------------------------------------------------------
    # TABS: LLAMA CHATBOT & NOVA SONIC VOICE
    # --------------------------------------------------------
    llama_tab, nova_tab = st.tabs([
        "💬 Llama 4 Document Chatbot",
        "🎙️ Amazon Nova 2 Sonic Live Voice"
    ])

    # ========================================================
    # TAB 1: LLAMA 4 CHATBOT (FULL CONVERSATION HISTORY)
    # ========================================================
    with llama_tab:
        col_title, col_clear = st.columns([4, 1])
        with col_title:
            st.subheader("💬 Chat with Llama 4")
            st.caption("Ask questions strictly grounded in your uploaded documents. All conversation turns are preserved.")
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.llama_messages = []
                st.rerun()

        # Render conversation history
        for msg in st.session_state.llama_messages:
            role = msg["role"]
            with st.chat_message(role):
                st.markdown(msg["content"])
                if role == "assistant" and msg.get("sources"):
                    with st.expander("📄 View Retrieved Document Sources"):
                        for doc, score in msg["sources"]:
                            st.markdown(f"• **{doc.metadata.get('source')}** (Page {doc.metadata.get('page')}) — *FAISS Distance: {score:.4f}*")
                        if msg.get("context"):
                            st.code(msg["context"])

        # Chat Input at bottom
        if user_prompt := st.chat_input("Ask a question about your uploaded documents..."):
            # 1. Append user message
            st.session_state.llama_messages.append({
                "role": "user",
                "content": user_prompt
            })
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # 2. Generate Llama 4 answer
            with st.chat_message("assistant"):
                with st.spinner("Llama 4 is analyzing documents..."):
                    try:
                        answer, results, context = answer_from_llama(user_prompt)
                        st.markdown(answer)

                        if results:
                            with st.expander("📄 View Retrieved Document Sources"):
                                for doc, score in results:
                                    st.markdown(f"• **{doc.metadata.get('source')}** (Page {doc.metadata.get('page')}) — *FAISS Distance: {score:.4f}*")
                                st.code(context)

                        st.session_state.llama_messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": results,
                            "context": context
                        })
                    except Exception as e:
                        st.error(f"Llama 4 error: {e}")

    # ========================================================
    # TAB 2: NOVA SONIC LIVE VOICE
    # ========================================================
    with nova_tab:
        st.subheader("🎙️ Amazon Nova 2 Sonic Live Voice")
        st.caption(
            "Real-time voice conversation with Amazon Nova 2 Sonic. "
            "Speak into your microphone — Nova Sonic answers in voice through your speakers and displays live text transcriptions."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "▶️ Start Voice Chat",
                use_container_width=True,
                disabled=st.session_state.nova_running
            ):
                start_nova()
                st.rerun()

        with col2:
            if st.button(
                "⏹️ End Session",
                use_container_width=True,
                disabled=not st.session_state.nova_running
            ):
                stop_nova()
                st.rerun()

        live_nova_ui()


if __name__ == "__main__":
    main()