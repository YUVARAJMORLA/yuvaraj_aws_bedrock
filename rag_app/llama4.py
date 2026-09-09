# converse()
import os
import boto3

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
)

model_id = os.environ.get(
    "LLAMA_MODEL_ID",
    (
        "arn:aws:bedrock:us-east-1:776817040428:"
        "inference-profile/us.meta.llama4-maverick-17b-instruct-v1:0"
    )
)


def ask_question(question: str, conversation_history: list | None = None) -> tuple[str, list]:
    if conversation_history is None:
        conversation_history = []

    conversation_history.append({
        "role": "user",
        "content": [{"text": question}]
    })

    response = bedrock.converse(
        modelId=model_id,
        messages=conversation_history,
        inferenceConfig={
            "maxTokens": 2048,
            "temperature": 0.5,
            "topP": 0.9
        },
        additionalModelRequestFields={}
    )

    response_text = response["output"]["message"]["content"][0]["text"]
    conversation_history.append({
        "role": "assistant",
        "content": [{"text": response_text}]
    })
    return response_text, conversation_history


def main():
    print("=" * 60)
    print("  Llama 4 Bedrock Terminal Assistant")
    print("  Ask any question. Type 'exit' or 'quit' to end.")
    print("  Type 'clear' to reset conversation context.")
    print("=" * 60)

    history = []
    while True:
        try:
            prompt = input("\nEnter your question: ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("Exiting...")
                break
            if prompt.lower() in ("clear", "reset"):
                history = []
                print("Conversation history cleared.")
                continue

            print("\nThinking...")
            response_text, history = ask_question(prompt, history)
            print(f"\nResponse:\n{response_text}")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
