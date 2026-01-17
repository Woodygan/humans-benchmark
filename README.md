# HUMANS Benchmark

**Authors:** Woody Haosheng Gan¹, William Held²'³, Diyi Yang²

¹University of Southern California, ²Stanford University, ³OpenAthena

**HUMANS: HUman-aligned Minimal Audio evaluatioN Subsets for Large Audio Models**

This repo is part of the **Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment** paper.

HUMANS Benchmark is designed to efficiently evaluate Large Audio Models using minimal subsets while predicting human preferences through learned regression weights.

---

## 🚀 Quick Start
```python
from HUMANS import HUMANSEvaluator, Message, ModelResponse

# Initialize evaluator
evaluator = HUMANSEvaluator(
    dataset_name="rma9248/humans-benchmark",
    subset="n50"
)

# Define your model's prediction function
def predict_fn(messages, audio_output, text_output, tools=None, tool_choice="auto"):
    # Your model inference code here
    return ModelResponse(text="response", audio_path=None, tool_calls=None)

# Run evaluation
results = evaluator.evaluate(predict_fn=predict_fn, mode="both")

print(f"Human Preference Score: {results['human_score']:.4f}")
print(f"Benchmark Score: {results['benchmark_score']:.4f}")
```

---

## 📦 Installation

### Option 1: Install via pip
```bash
pip install git+https://github.com/Woodygan/humans-benchmark.git
```

### Option 2: Clone and install in editable mode
```bash
git clone https://github.com/Woodygan/humans-benchmark.git
cd humans-benchmark
pip install -e .
```

### Requirements

- **Python 3.8+**
- **OpenAI API key** (required): Used for LLM-based metrics across all tasks. Cost: ~$0.1 per evaluation
- **Google API key** (optional): Used for SpeakBench evaluation following the original framework. If not provided, falls back to OpenAI models.

### Setting up API keys

**Option 1: Using a `.env` file (recommended)**

Create a `.env` file in your project directory:
```bash
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here  # Optional
```

**Option 2: Using export in your shell**
```bash
export OPENAI_API_KEY='your-openai-api-key-here'
export GOOGLE_API_KEY='your-google-api-key-here'  # Optional
```

---

## 📂 Repository Structure
```
humans-benchmark/
├── src/
│   └── HUMANS/
│       ├── __init__.py          # Package exports
│       ├── evaluator.py          # Main HUMANSEvaluator class
│       ├── message.py            # Message and ModelResponse dataclasses
│       ├── whisper.py            # Whisper transcription utilities
│       ├── cava/                 # CAVA benchmark metrics
│       │   └── metric.py
│       ├── dynamic_superb/       # Dynamic Superb metrics
│       │   └── metric.py
│       ├── speakbench/           # SpeakBench metrics
│       │   └── metric.py
│       ├── ultraeval/            # UltraEval metrics
│       │   └── metric.py
│       └── wildspeech/           # WildSpeech metrics
│           └── metric.py
├── example/
│   └── example.py                # Complete working example
├── requirements.txt              # Package dependencies
├── pyproject.toml               # Package configuration
├── setup.py                     # Setup script
└── README.md                    # This file
```

---

## 🎯 Running the Example

We provide a complete working example that demonstrates how to evaluate GPT-4o Audio Preview on the HUMANS benchmark.
```bash
# Navigate to the example directory
cd example

# Run the example
python example.py
```

The example demonstrates:
- Initializing the HUMANSEvaluator
- Implementing a prediction function for OpenAI's GPT-4o Audio Preview
- Handling audio input/output
- Supporting function calling tasks
- Evaluating the model and saving results

**Note:** Make sure you have set your `OPENAI_API_KEY` environment variable before running the example.

---

## 📖 API Documentation

### HUMANSEvaluator

#### Initialization
```python
evaluator = HUMANSEvaluator(
    dataset_name: str = "rma9248/humans-benchmark",
    subset: str = "n50",
    cache_dir: Optional[str] = None,
    audio_dir: str = "humans-audio",
    delete_audio_on_cleanup: bool = False
)
```

**Parameters:**

- `dataset_name` (str): HuggingFace dataset identifier
  - Default: `"rma9248/humans-benchmark"`

- `subset` (str): Evaluation subset to use
  - Options: `"n10"`, `"n20"`, `"n30"`, `"n50"`, `"n100"`, `"n200"`
  - Default: `"n50"`
  - Larger subsets provide more accurate evaluation but take longer

- `cache_dir` (Optional[str]): Directory to cache the downloaded dataset
  - Default: `None` (uses HuggingFace default cache)

- `audio_dir` (str): Directory to save temporary audio files during evaluation
  - Default: `"humans-audio"`
  - Audio files are stored here for processing by metrics

- `delete_audio_on_cleanup` (bool): Whether to automatically delete audio directory when evaluator is destroyed
  - Default: `False`
  - Set to `True` to automatically clean up audio files after evaluation

#### Evaluation
```python
results = evaluator.evaluate(
    predict_fn: Callable,
    mode: str = "both",
    save_results: bool = True,
    results_path: Optional[str] = None,
    verbose: bool = True
)
```

**Parameters:**

- `predict_fn` (Callable): Your model's prediction function (see below for detailed specification)
  - **Required**
  - Function signature: `predict_fn(messages, audio_output, text_output, tools=None, tool_choice="auto") -> ModelResponse`

- `mode` (str): Evaluation mode
  - `"human"`: Compute human preference score only (0-1 scale)
  - `"benchmark"`: Compute full benchmark score approximation
  - `"both"`: Compute both scores (default)

- `save_results` (bool): Whether to save results to a JSON file
  - Default: `True`

- `results_path` (Optional[str]): Path to save the results JSON file
  - Default: `None` (auto-generates filename with timestamp: `humans_results_YYYYMMDD_HHMMSS.json`)

- `verbose` (bool): Show progress bar and logging during evaluation
  - Default: `True`

**Returns:**

A dictionary containing:
```python
{
    "human_score": 0.75,              # Human preference score [0, 1] (if mode="human" or "both")
    "benchmark_score": 0.68,           # Full benchmark score (if mode="benchmark" or "both")
    "num_items": 50,                   # Number of evaluation items
    "subset": "n50",                   # Subset used
    "audio_dir": "/path/to/audio",     # Directory containing audio files
    "results_path": "/path/to/results.json",  # Path to saved results (if save_results=True)
    "details": [                       # Per-item evaluation details
        {
            "item_id": "item_001",
            "task": "speech_recognition",
            "dataset": "dynamic_superb",
            "metric": "word_error_rate",
            "score": 0.85,
            "audio_output_expected": False,
            "text_output_expected": True,
            "latency": 1.23,           # Response time in seconds
            "metadata": {              # Task-specific metadata
                "error_type": None,
                "reference": "ground truth text"
            }
        },
        # ... more items
    ]
}
```

### Prediction Function Interface

#### predict_fn Specification

Your `predict_fn` must implement the following interface:
```python
def predict_fn(
    messages: List[Message],
    audio_output: bool,
    text_output: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto"
) -> ModelResponse:
    """
    Model prediction function for HUMANS benchmark.

    Args:
        messages: List of conversation messages (Message objects)
        audio_output: Whether the task expects audio output
        text_output: Whether the task expects text output
        tools: Optional list of tool/function definitions for function calling tasks
        tool_choice: Tool choice strategy - "auto", "required", or "none"

    Returns:
        ModelResponse object with model outputs
    """
    # Your model inference logic here
    pass
```

#### Input: messages

A list of `Message` objects representing the conversation history:
```python
@dataclass
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    text_input: Optional[str] = None           # Text content
    audio_path: Optional[str] = None           # Path to audio file (.wav)
    tool_calls: Optional[List[Dict]] = None    # Function calls from assistant (OpenAI format)
    tool_call_id: Optional[str] = None         # ID matching the tool call (for OpenAI API models)
    name: Optional[str] = None                 # Function name (for Gemini and other models)
```

**Field Descriptions:**

- `tool_call_id`: Used in tool response messages to match back to the original function call. Required for OpenAI API models (matches the `"id"` field from the assistant's tool_calls)
- `name`: Function name used in tool response messages. Required for models like Gemini that identify function responses by name instead of ID

**Message Examples:**
```python
# User message with text only
Message(role="user", text_input="What is the weather?")

# User message with audio input
Message(role="user", text_input="Transcribe this:", audio_path="/path/to/audio.wav")

# Assistant message with tool calls (OpenAI format)
Message(role="assistant", text_input="Let me check the weather",
        tool_calls=[{
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "San Francisco"}  # Dictionary, not JSON string!
            }
        }])

# Tool response message (includes both tool_call_id and name for compatibility)
Message(role="tool", text_input="Sunny, 72°F",
        tool_call_id="call_123",  # For OpenAI models
        name="get_weather")        # For Gemini and similar models
```

#### Input: audio_output and text_output

These boolean flags indicate what type of output the task expects:

- `audio_output=True`: Task requires audio response (e.g., speech synthesis, voice conversion)
- `text_output=True`: Task requires text response (e.g., speech recognition, classification)
- Both can be `True` for tasks requiring both modalities

#### Input: tools and tool_choice

For function calling tasks, the benchmark provides tool definitions and expects responses in **OpenAI API compatible format**.

- `tools`: List of available function definitions following **OpenAI function calling format**:
```python
  [
      {
          "type": "function",
          "function": {
              "name": "function_name",
              "description": "Function description",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "param1": {"type": "string", "description": "..."},
                      # ... more parameters
                  },
                  "required": ["param1"]
              }
          }
      }
  ]
```

  **Note:** This format is compatible with OpenAI API. If your model uses a different format (e.g., Google's function calling format), you'll need to convert between formats in your `predict_fn`.

- `tool_choice`: Strategy for function calling (OpenAI API compatible)
  - `"auto"`: Model decides whether to call functions
  - `"required"`: Model must call at least one function
  - `"none"`: Model should not call functions

#### Output: ModelResponse

Return a `ModelResponse` object:
```python
@dataclass
class ModelResponse:
    text: str                                  # Text output (required, use "" if none)
    audio_path: Optional[str] = None           # Path to generated audio file (.wav)
    tool_calls: Optional[List[Dict]] = None    # Function calls (see format below)
    metadata: Optional[Dict] = None            # Optional metadata
```

**Function Call Format (IMPORTANT - READ CAREFULLY):**

When your model calls functions, return them in **OpenAI API compatible format**. This is a specific format that you must follow exactly:
```python
tool_calls = [
    {
        "id": "call_abc123",              # Unique call ID (optional) - used by your model to match tool
                                           # responses back to the original call (required in
                                           # OpenAI API). Some models like Gemini use the function
                                           # name instead of ID for matching.

        "type": "function",                # Always "function" (required)

        "function": {
            "name": "function_name",       # Function name (string) - also used for matching tool
                                           # responses in models like Gemini

            "arguments": {                 # Arguments as a DICTIONARY (NOT a JSON string!)
                "param1": "value1",        # Each argument as a key-value pair
                "param2": 42
            }
        }
    }
]
```


**Important Notes:**

- **For OpenAI models:** The `"id"` field is used to match tool responses back to the original function call
- **For Google Gemini and similar models:** The `"name"` field is used for matching instead of `"id"`. We include both fields to support different model architectures
- **Arguments format:** The `"arguments"` field MUST be a Python dictionary, NOT a JSON string. If your model API returns arguments as a JSON string (like OpenAI does), parse it with `json.loads()` before returning
- **Multi-turn function calling:** The evaluator automatically handles the conversation loop - you don't need to implement this yourself
- **Function responses:** The evaluator provides function responses for testing purposes

---

## 📊 Dataset Structure

The benchmark evaluates models across multiple datasets and tasks:

**Dataset Items Include:**
- `item_id`: Unique identifier
- `task`: Task name (e.g., "speech_recognition", "emotion", "function_calling")
- `dataset`: Source dataset name
- `metric`: Evaluation metric used
- `audio_input`: Input audio (if applicable)
- `text_input`: Input text prompt (if applicable)
- `audio_reference`: Reference/ground truth audio (if applicable)
- `text_reference`: Reference/ground truth text in list format (e.g., `text_reference[0]` for single answer)
- `audio_output`: Whether task expects audio output
- `text_output`: Whether task expects text output
- `human_preference_weight`: Weight for human preference regression
- `full_benchmark_weight`: Weight for full benchmark score

**Available Subsets:**
- `n10`: 10 evaluation items (fast, less accurate)
- `n20`: 20 evaluation items
- `n30`: 30 evaluation items
- `n50`: 50 evaluation items (default)
- `n100`: 100 evaluation items
- `n200`: 200 evaluation items

---

## 🔧 Cleanup

To manually delete the audio directory after evaluation:
```python
evaluator.cleanup_audio()
```

Or set `delete_audio_on_cleanup=True` during initialization for automatic cleanup.

---

## 📜 License

MIT License

---

## 🔗 Links

- **Dataset:** [HuggingFace Dataset](https://huggingface.co/datasets/rma9248/humans-benchmark)
- **Repository:** [GitHub](https://github.com/Woodygan/humans-benchmark)
- **Issues:** [Bug Tracker](https://github.com/Woodygan/humans-benchmark/issues)

---

## 📚 Citation
```bibtex
@misc{gan2026humans,
  title={Putting HUMANS first: Efficient LAM Evaluation with Human Preference Alignment},
  author={Gan, Woody Haosheng and Held, William and Yang, Diyi},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/rma9248/humans-benchmark}},
  note={A benchmark for efficiently evaluating Large Audio Models using minimal subsets aligned with human preferences}
}
```

This benchmark builds upon several existing audio evaluation frameworks:
```bibtex
@misc{cava2025,
  title = {CAVA: Comprehensive Assessment of Voice Assistants},
  author = {Held, Will and Ryan, Michael J. and Shrivastava, Aditya and Khan, Ali Sartaz and Ziems, Caleb and Li, Ella and Bartelds, Martijn and Sun, Michael and Li, Tan and Gan, Woody and Yang, Diyi},
  year = {2025},
  url = {https://talkarena.org/cava},
  howpublished = {\url{https://github.com/SALT-NLP/CAVA}},
  note = {A benchmark for evaluating large audio models (LAMs) capabilities across six domains: turn taking, instruction following, function calling, tone awareness, safety, and latency}
}

@article{huang2024dynamic,
  title={Dynamic-superb phase-2: A collaboratively expanding benchmark for measuring the capabilities of spoken language models with 180 tasks},
  author={Huang, Chien-yu and Chen, Wei-Chih and Yang, Shu-wen and Liu, Andy T and Li, Chen-An and Lin, Yu-Xiang and Tseng, Wei-Cheng and Diwan, Anuj and Shih, Yi-Jen and Shi, Jiatong and others},
  journal={arXiv preprint arXiv:2411.05361},
  year={2024}
}

@article{he2024ultraeval,
  title={Ultraeval: A lightweight platform for flexible and comprehensive evaluation for llms},
  author={He, Chaoqun and Luo, Renjie and Hu, Shengding and Zhao, Yuanqian and Zhou, Jie and Wu, Hanghao and Zhang, Jiajie and Han, Xu and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:2404.07584},
  year={2024}
}

@article{manakul2025audiojudge,
  title={AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation},
  author={Manakul, Potsawee and Gan, Woody Haosheng and Ryan, Michael J and Khan, Ali Sartaz and Sirichotedumrong, Warit and Pipatanakul, Kunat and Held, William and Yang, Diyi},
  journal={arXiv preprint arXiv:2507.12705},
  year={2025}
}

@article{zhang2025wildspeech,
  title={WildSpeech-Bench: Benchmarking Audio LLMs in Natural Speech Conversation},
  author={Zhang, Jian and Zhang, Linhao and Lei, Bokai and Wu, Chuhan and Jia, Wei and Zhou, Xiao},
  journal={arXiv preprint arXiv:2506.21875},
  year={2025}
}
```