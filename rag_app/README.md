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


## Applications

This repository contains three separate AWS Bedrock applications. They are independent of each other and can be run separately.

### `app.py` — RAG Application

`app.py` is the **RAG (Retrieval-Augmented Generation) application**. It uses AWS Bedrock along with a knowledge source to retrieve relevant information and generate responses based on the retrieved content.

This application can be run independently and does not depend on `llama.py` or `nova.py`.

### `llama.py` — Llama Chat Application

`llama.py` is a **standalone chat application** using Meta Llama models through AWS Bedrock.

It is a normal conversational application and is **not connected to the RAG application in `app.py`**. It can be run independently.

The implementation uses the newer AWS Bedrock APIs and approaches rather than following the original project's `InvokeModel` implementation directly.

### `nova.py` — Amazon Nova Chat Application

`nova.py` is another **standalone chat application**, this time using Amazon Nova through AWS Bedrock.

It is also independent of `app.py` and can be run as a separate application.

In short:

```text
app.py
   └── RAG Application
       └── Can run independently


llama.py
   └── Llama Chat Application
       └── Can run independently


nova.py
   └── Amazon Nova Chat Application
       └── Can run independently
```

## Reference

This project was developed independently using AWS Bedrock documentation, tutorials, and open-source projects as references for learning and implementation ideas.

The AWS Bedrock project by **Krish Naik** was used as one of the references while learning AWS Bedrock concepts and implementation approaches:

* GitHub: https://github.com/krishnaik06/AWS-Bedrock
* YouTube Playlist: https://youtube.com/playlist?list=PLZoTAELRMXVP5zpBfH7pab4aB1LbmCM1z

The original GitHub repository is licensed under **GPL-3.0**.

This implementation has been substantially rewritten and adapted, including changes to the application structure, models, AWS Bedrock APIs, and overall implementation. It also uses newer approaches such as the **Converse API** for supported models instead of directly following the original `InvokeModel` approach.

The references above are included for transparency and to acknowledge the resources used during the learning and development process.
