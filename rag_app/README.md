# Multimodal RAG & Voice Assistant with AWS Bedrock

An advanced Retrieval-Augmented Generation (RAG) and Voice AI application powered by **AWS Bedrock**, featuring:
- **Multimodal Document RAG & Chatbot** (`app.py` via Streamlit): Upload PDF, DOCX, PPTX, XLSX, and text documents, index them using **Titan Text Embeddings** + **FAISS**, and chat using **Meta Llama 4** on Bedrock.
- **Real-Time Voice Assistant** (`nova.py` & in `app.py`): Low-latency bidirectional speech-to-speech interaction powered by **Amazon Nova 2 Sonic** streaming over WebSockets/event stream with PyAudio.
- **Terminal Llama Assistant** (`llama4.py`): Fast, lightweight command-line conversational agent powered by Bedrock Converse API.

---

## Architecture & Features

- **Embeddings**: `amazon.titan-embed-text-v1`
- **Text LLM**: Meta Llama 4 (`us.meta.llama4-maverick-17b-instruct-v1:0` inference profile)
- **Voice LLM**: Amazon Nova 2 Sonic (`amazon.nova-2-sonic-v1:0`)
- **Vector Store**: FAISS (in-memory / local CPU)
- **Frameworks**: Streamlit, LangChain, AWS Boto3 & Bedrock Runtime SDK

---

## Prerequisites

1. **Python 3.12.+** installed.
2. **AWS Account** with model access enabled in AWS Bedrock for:
   - Amazon Titan Embeddings G1 - Text
   - Meta Llama 4 (or your configured inference profile)
   - Amazon Nova 2 Sonic
3. **Microphone & Speaker** (for voice interaction). On Windows, `PyAudio` installs directly via pip. On Linux, ensure `portaudio19-dev` is installed (`sudo apt-get install portaudio19-dev`).

---

## Quickstart Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/rag_app.git
cd rag_app
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure AWS Credentials
You can configure your AWS credentials using the AWS CLI:
```bash
aws configure
```
Or copy `.env.example` to `.env` and configure your environment variables:
```bash
# Windows PowerShell
$env:AWS_ACCESS_KEY_ID="your-access-key-id"
$env:AWS_SECRET_ACCESS_KEY="your-secret-access-key"
$env:AWS_DEFAULT_REGION="us-east-1"
```

---

## Running the Applications

### 1. Web Application (RAG Chatbot + Voice)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Live Nova Sonic Voice Assistant (Terminal)
```bash
python nova.py
```
Speak into your microphone and hear responses generated in real-time. Press **Enter** to stop the session.

### 3. Llama 4 Conversational Terminal Assistant
```bash
python llama4.py
```


