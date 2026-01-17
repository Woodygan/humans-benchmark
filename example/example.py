import os
import base64
from typing import List, Optional, Dict, Any
from openai import OpenAI
import json
from HUMANS import HUMANSEvaluator, Message, ModelResponse
from dotenv import load_dotenv
load_dotenv()
# Initialize OpenAI client
# Note: GOOGLE_API_KEY is also supported for SpeakBench tasks (optional)
# Set API keys via .env file or: export OPENAI_API_KEY='your-key'
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def predict_fn(
    messages: List[Message],
    audio_output: bool,
    text_output: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto"
) -> ModelResponse:
    """
    Model prediction function using GPT-4o Audio Preview
    """
    # Convert HUMANS messages to OpenAI format
    openai_messages = []

    for msg in messages:
        # Handle tool messages
        if msg.role == "tool":
            openai_messages.append({
                "role": "tool",
                "content": msg.text_input,
                "tool_call_id": msg.tool_call_id
            })
            continue

        # Build content for regular messages
        content = []

        if msg.text_input:
            content.append({"type": "text", "text": msg.text_input})

        if msg.audio_path:
            with open(msg.audio_path, "rb") as f:
                encoded_audio = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "input_audio",
                "input_audio": {"data": encoded_audio, "format": "wav"}
            })

        message = {"role": msg.role, "content": content}

        # Add tool calls if present
        if msg.tool_calls:
            formatted_tool_calls = []
            for tc in msg.tool_calls:
                formatted_tool_calls.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": json.dumps(tc["function"]["arguments"])
                    }
                })
            message["tool_calls"] = formatted_tool_calls

        openai_messages.append(message)

    # Prepare API call
    api_args = {
        "model": "gpt-4o-audio-preview",
        "messages": openai_messages,
        "temperature": 0.8,
    }

    # Add audio modality if needed
    if audio_output:
        api_args["modalities"] = ["text", "audio"]
        api_args["audio"] = {"voice": "alloy", "format": "wav"}

    # Add tools if provided
    if tools is not None:
        api_args["tools"] = tools
        api_args["tool_choice"] = tool_choice

    # Make API call
    completion = client.chat.completions.create(**api_args)
    message = completion.choices[0].message

    response_text = message.content or "" if text_output else None
    response_audio_path = None
    response_tool_calls = None

    # Extract tool calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        response_tool_calls = []
        for tool_call in message.tool_calls:
            response_tool_calls.append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
            })

    # Extract audio output
    if audio_output and hasattr(message, "audio") and message.audio:
        if hasattr(message.audio, "transcript") and message.audio.transcript and text_output:
            response_text = message.audio.transcript

        if hasattr(message.audio, "data") and message.audio.data:
            import tempfile
            audio_data = base64.b64decode(message.audio.data)
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.wav', delete=False) as f:
                f.write(audio_data)
                response_audio_path = f.name

    return ModelResponse(
        text=response_text,
        audio_path=response_audio_path,
        tool_calls=response_tool_calls
    )

# Initialize evaluator
evaluator = HUMANSEvaluator(
    dataset_name="rma9248/humans-benchmark",
    subset="n10",
    audio_dir="humans-audio",
    delete_audio_on_cleanup=False
)

# Run evaluation
results = evaluator.evaluate(
    predict_fn=predict_fn,
    mode="both",
    save_results=True,
    verbose=True
)

# Print results
print(f"Human Preference Score: {results['human_score']:.4f}")
print(f"Benchmark Score: {results['benchmark_score']:.4f}")
print(f"Number of Items: {results['num_items']}")
print(f"Results saved to: {results['results_path']}")
