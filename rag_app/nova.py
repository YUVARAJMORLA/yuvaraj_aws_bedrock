import os
import asyncio
import base64
import json
import uuid
import boto3
import pyaudio
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver


def ensure_aws_credentials(default_region="us-east-1"):
    """
    Automatically retrieves AWS credentials configured via `aws configure`
    (stored in ~/.aws/credentials and ~/.aws/config) using boto3 and populates
    the environment variables for Smithy's EnvironmentCredentialsResolver.
    This eliminates the need to hardcode access keys in the code.
    """
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
        
        region = session.region_name or default_region
        if "AWS_DEFAULT_REGION" not in os.environ:
            os.environ["AWS_DEFAULT_REGION"] = region
        return region
    except Exception as e:
        print(f"Warning: Could not auto-load credentials from AWS configuration: {e}")
        return default_region

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = pyaudio.paInt16

CHUNK_SIZE = 1024

# AUDIO FIX: use a slightly larger playback chunk
OUTPUT_CHUNK_SIZE = 2048


class SimpleNovaSonic:
    def __init__(self, model_id='amazon.nova-2-sonic-v1:0', region=None):
        configured_region = ensure_aws_credentials()
        self.model_id = model_id
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", configured_region)
        self.client = None
        self.stream = None
        self.response = None
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.audio_queue = asyncio.Queue()
        self.role = None
        self.display_assistant_text = False
        
    def _initialize_client(self):
        ensure_aws_credentials(self.region)
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.client = BedrockRuntimeClient(config=config)
    
    async def send_event(self, event_json):
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(
                bytes_=event_json.encode('utf-8')
            )
        )
        await self.stream.input_stream.send(event)
    
    async def start_session(self):
        if not self.client:
            self._initialize_client()

        print("\nConnecting to Amazon Nova Sonic...")

        self.stream = await self.client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(
                model_id=self.model_id
            )
        )

        self.is_active = True
        
        session_start = '''
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }
        '''

        await self.send_event(session_start)
        
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
            "You are a warm, professional, and helpful male AI assistant. "
            "Give accurate answers that sound natural, direct, and human. "
            "Start by answering the user's question clearly in 1–2 sentences. "
            "Then, expand only enough to make the answer understandable, "
            "staying within 3–5 short sentences total. "
            "Avoid sounding like a lecture or essay."
        )
        
        text_input = f'''
        {{
            "event": {{
                "textInput": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.content_name}",
                    "content": "{system_prompt}"
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
        
        self.response = asyncio.create_task(
            self._process_responses()
        )
    
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
    
    async def end_session(self):
        if not self.is_active:
            return

        prompt_end = f'''
        {{
            "event": {{
                "promptEnd": {{
                    "promptName": "{self.prompt_name}"
                }}
            }}
        }}
        '''

        await self.send_event(prompt_end)
        
        session_end = '''
        {
            "event": {
                "sessionEnd": {}
            }
        }
        '''

        await self.send_event(session_end)

        await self.stream.input_stream.close()
    
    async def _process_responses(self):
        try:
            while self.is_active:

                output = await self.stream.await_output()
                result = await output[1].receive()
                
                if result.value and result.value.bytes_:

                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)
                    
                    if 'event' in json_data:

                        if 'contentStart' in json_data['event']:

                            content_start = json_data['event']['contentStart']

                            self.role = content_start['role']

                            if 'additionalModelFields' in content_start:

                                additional_fields = json.loads(
                                    content_start['additionalModelFields']
                                )

                                if additional_fields.get(
                                    'generationStage'
                                ) == 'SPECULATIVE':

                                    self.display_assistant_text = True

                                else:
                                    self.display_assistant_text = False
                                
                        elif 'textOutput' in json_data['event']:

                            text = json_data['event']['textOutput']['content']

                            if (
                                self.role == "ASSISTANT"
                                and self.display_assistant_text
                            ):
                                print(f"Assistant: {text}")

                            elif self.role == "USER":
                                print(f"User: {text}")
                        
                        elif 'audioOutput' in json_data['event']:

                            audio_content = (
                                json_data['event']['audioOutput']['content']
                            )

                            audio_bytes = base64.b64decode(audio_content)

                            # AUDIO FIX:
                            # Put the complete received audio packet
                            # into the queue without modifying it.
                            await self.audio_queue.put(audio_bytes)

        except Exception as e:
            print(f"Error processing responses: {e}")
    
    async def play_audio(self):

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,

            # AUDIO FIX:
            # Larger hardware buffer reduces playback underflow.
            frames_per_buffer=2048
        )

        try:

            while self.is_active:

                audio_data = await self.audio_queue.get()

                # AUDIO FIX:
                # Play larger pieces instead of 1024-byte pieces.
                for i in range(
                    0,
                    len(audio_data),
                    OUTPUT_CHUNK_SIZE
                ):

                    if not self.is_active:
                        break

                    end = min(
                        i + OUTPUT_CHUNK_SIZE,
                        len(audio_data)
                    )

                    chunk = audio_data[i:end]

                    # AUDIO FIX:
                    # Do not block the asyncio event loop while
                    # PyAudio writes audio to the speaker.
                    await asyncio.to_thread(
                        stream.write,
                        chunk
                    )

                    # AUDIO FIX:
                    # Removed asyncio.sleep(0.001)
                    # because it can introduce tiny playback gaps.

        except Exception as e:
            print(f"Error playing audio: {e}")

        finally:

            stream.stop_stream()
            stream.close()
            p.terminate()

            print("Audio playing stopped.")

    async def capture_audio(self):

        p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print(
            "Starting audio capture. Speak into your microphone..."
        )

        print("Press Enter to stop...")
        
        await self.start_audio_input()
        
        try:

            while self.is_active:

                # AUDIO FIX:
                # Microphone read is blocking, so move it
                # outside the asyncio event loop.
                audio_data = await asyncio.to_thread(
                    stream.read,
                    CHUNK_SIZE,
                    exception_on_overflow=False
                )

                await self.send_audio_chunk(audio_data)

                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"Error capturing audio: {e}")

        finally:

            stream.stop_stream()
            stream.close()
            p.terminate()

            print("Audio capture stopped.")

            await self.end_audio_input()


async def main():

    nova_client = SimpleNovaSonic()

    await nova_client.start_session()
    
    playback_task = asyncio.create_task(
        nova_client.play_audio()
    )

    capture_task = asyncio.create_task(
        nova_client.capture_audio()
    )
    
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            input
        )
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        tasks = []

        if not playback_task.done():
            tasks.append(playback_task)

        if not capture_task.done():
            tasks.append(capture_task)

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(
                *tasks,
                return_exceptions=True
            )
        
        await nova_client.end_session()

        nova_client.is_active = False

        if (
            nova_client.response
            and not nova_client.response.done()
        ):
            nova_client.response.cancel()

        print("Session ended")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")