from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, Callable
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tempfile
import os
from pathlib import Path
import soundfile as sf
import shutil
import time
from datetime import datetime
from .message import Message, ModelResponse
from .dynamic_superb.metric import llm_classification, pos_estimation, phoneme_error_rate, word_error_rate
from .ultraeval.metric import ultraeval_gpt_score, ultraeval_exist_match
from .wildspeech.metric import wildspeech_gpt_score, wildspeech_utmos
from .speakbench.metric import speakbench_WinRate
from .cava.metric import (
    emotion_match, speaker_diarization_jer, multimodal_instruction_following,
    deception_detection_exact_match, jailbreak_refusal_rate, jeopardy_correctness_pedant,
    jeopardy_latency_score, pronunciation_match, function_calling_match
)
from .whisper import WhisperTranscriber

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
# @dataclass
# class Message:
#     """Message in a conversation"""
#     role: Literal["user", "assistant", "system", "tool"]
#     text_input: Optional[str] = None
#     audio_path: Optional[str] = None
#     tool_calls: Optional[List[Dict[str, Any]]] = None
#     tool_call_id: Optional[str] = None
#     name: Optional[str] = None


# @dataclass
# class ModelResponse:
#     """Response from a model"""
#     text: str
#     audio_path: Optional[str] = None
#     tool_calls: Optional[List[Dict[str, Any]]] = None
#     metadata: Optional[Dict[str, Any]] = field(default_factory=dict)


class HUMANSEvaluator:
    """
    HUMANS Benchmark evaluator
    
    Efficiently evaluates Large Audio Models using minimal subsets while
    predicting human preferences through learned regression weights.
    """
    
    def __init__(
        self,
        dataset_name: str = "rma9248/humans-benchmark",
        subset: str = "n50",
        cache_dir: Optional[str] = None,
        audio_dir: str = "humans-audio",
        delete_audio_on_cleanup: bool = False,
    ):
        """
        Initialize HUMANS Benchmark
        
        Args:
            dataset_name: HuggingFace dataset name
            subset: Subset to use (n10, n20, n30, n50, n100, n200)
            cache_dir: Cache directory for dataset
            audio_dir: Directory to save audio files (default: "humans-audio")
            delete_audio_on_cleanup: Whether to delete audio directory on cleanup (default: False)
        """
        print(f"Loading HUMANS Benchmark: {dataset_name} ({subset})...")
        self.dataset_name = dataset_name
        self.subset = subset
        self.delete_audio_on_cleanup = delete_audio_on_cleanup
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset(dataset_name, split=subset, cache_dir=cache_dir)
        self.items = list(self.dataset)
        
        print(f"✓ Loaded {len(self.items)} evaluation items")
        
        # Extract weights and bias
        self.human_weights = np.array([item['human_preference_weight'] for item in self.items])
        self.benchmark_weights = np.array([item['full_benchmark_weight'] for item in self.items])
        self.regression_bias = self.items[0]['human_regression_bias']
        
        # Create persistent directory for audio files
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Audio directory to save temp audio file: {self.audio_dir.absolute()}")
        
        # Check for OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it either:\n"
                "  1. In a .env file: OPENAI_API_KEY=your-key\n"
                "  2. In your environment: export OPENAI_API_KEY='your-key'"
            )
        
        # Initialize Whisper transcriber for UltraEval metrics
        print(f"✓ Initializing Whisper transcriber...")
        self.whisper_transcriber = WhisperTranscriber()
        
        print(f"✓ Ready to evaluate")
    
    def __del__(self):
        """Clean up audio directory if delete_audio_on_cleanup is True"""
        if self.delete_audio_on_cleanup and hasattr(self, 'audio_dir') and self.audio_dir.exists():
            print(f"Cleaning up audio directory: {self.audio_dir}")
            shutil.rmtree(self.audio_dir, ignore_errors=True)
    
    def cleanup_audio(self):
        """Manually delete the audio directory"""
        if hasattr(self, 'audio_dir') and self.audio_dir.exists():
            print(f"Deleting audio directory: {self.audio_dir}")
            shutil.rmtree(self.audio_dir, ignore_errors=True)
            print("✓ Audio directory deleted")
        else:
            print("Audio directory does not exist or was already deleted")
    
    def evaluate(
        self,
        predict_fn: Callable[[List[Message], bool, bool], ModelResponse],
        mode: str = "both",
        save_results: bool = True,
        results_path: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on HUMANS Benchmark
        
        Args:
            predict_fn: Function (messages, audio_output, text_output) -> ModelResponse
                       Takes conversation messages and output flags, returns model response
            mode: "human" for human preference prediction (0-1 scale)
                  "benchmark" for full benchmark score approximation
                  "both" for both scores (default)
            save_results: Whether to save results to JSON file (default: True)
            results_path: Path to save results JSON. If None, uses "humans_results_{timestamp}.json"
            verbose: Show progress bar during evaluation
            
        Returns:
            Dictionary with evaluation results:
            - human_score (if mode="human" or "both"): Human preference score [0, 1]
            - benchmark_score (if mode="benchmark" or "both"): Full benchmark score
            - num_items: Number of items evaluated
            - details: Per-item results
        """
        if mode not in ["human", "benchmark", "both"]:
            raise ValueError(f"mode must be 'human', 'benchmark', or 'both', got '{mode}'")
        
        scores = []
        details = []
        
        iterator = tqdm(self.items, desc="Evaluating") if verbose else self.items
        
        for item in iterator:
            try:
                score, detail = self._eval_item(item, predict_fn)
                scores.append(score)
                details.append(detail)
            except Exception as e:
                if verbose:
                    print(f"\n⚠ Error on item {item['item_id']}: {e}")
                scores.append(0.0)
                details.append({
                    'item_id': item['item_id'],
                    'task': item.get('task', ''),
                    'error': str(e),
                    'score': 0.0,
                })
        
        scores = np.array(scores)
        
        # Compute scores based on mode
        results = {
            "num_items": len(scores),
            "subset": self.subset,
            "audio_dir": str(self.audio_dir.absolute()),
            "details": details,
        }
        
        if mode in ["human", "both"]:
            # Human preference score with regression, clipped to [0, 1]
            human_score_raw = np.dot(self.human_weights, scores) + self.regression_bias
            human_score = float(np.clip(human_score_raw, 0.0, 1.0))
            results["human_score"] = human_score
        
        if mode in ["benchmark", "both"]:
            # Benchmark score (weighted average)
            benchmark_score = float(np.dot(self.benchmark_weights, scores))
            results["benchmark_score"] = benchmark_score
        
        # Save results to JSON if requested
        if save_results:
            if results_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = f"humans_results_{timestamp}.json"
            
            results_path = Path(results_path)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            if verbose:
                print(f"\n✓ Results saved to: {results_path.absolute()}")
            
            results["results_path"] = str(results_path.absolute())
        
        return results
    
    def _save_audio_to_temp(self, audio_data: Dict[str, Any], prefix: str) -> Optional[str]:
        """
        Save audio data to persistent directory
        
        Args:
            audio_data: Dict with 'array' and 'sampling_rate' keys
            prefix: Prefix for filename
            
        Returns:
            Path to saved audio file or None if audio_data is None
        """
        if audio_data is None:
            return None
        audio_file = self.audio_dir / f"{prefix}.wav"
        sf.write(str(audio_file), audio_data['array'], audio_data['sampling_rate'])
        return str(audio_file)
    
    def _eval_item(self, item: Dict[str, Any], predict_fn: Callable) -> tuple:
        """
        Evaluate a single benchmark item
        
        Args:
            item: Benchmark item from dataset
            predict_fn: Model prediction function
            
        Returns:
            Tuple of (score, detail_dict)
        """
        item_id = item['item_id']
        
        # Save audio input to file if present
        audio_input_path = None
        if item['audio_input'] is not None:
            audio_input_path = self._save_audio_to_temp(
                item['audio_input'], 
                f"input_{item_id}"
            )
        
        # Save audio reference to file if present (needed for evaluation)
        audio_reference_path = None
        if item['audio_reference'] is not None:
            audio_reference_path = self._save_audio_to_temp(
                item['audio_reference'],
                f"ref_{item_id}"
            )
        
        # Get text input from item
        text_input = item.get('text_input')
        
        # Create initial message
        initial_message = Message(
            role="user",
            text_input=text_input,
            audio_path=audio_input_path
        )
        
        
        # Special handling for CAVA function_calling task - multi-turn conversation
        if item['dataset'] == 'cava' and item['task'] == 'function_calling':
            return self._eval_function_calling_item(
                item=item,
                initial_message=initial_message,
                predict_fn=predict_fn,
                audio_input_path=audio_input_path,
                audio_reference_path=audio_reference_path
            )
        # Standard single-turn evaluation for all other tasks
        messages = [initial_message]
        
        # Get model prediction with latency tracking
        latency = None
        try:
            start_time = time.time()
            response = predict_fn(
                messages,
                audio_output=item['audio_output'],
                text_output=item['text_output']
            )
            end_time = time.time()
            latency = end_time - start_time
        except Exception as e:
            # Model prediction failed
            detail = {
                'item_id': item_id,
                'task': item['task'],
                'dataset': item['dataset'],
                'metric': item['metric'],
                'score': 0.0,
                'audio_output_expected': item['audio_output'],
                'text_output_expected': item['text_output'],
                'metadata': {
                    'error': f"Model prediction failed: {str(e)}",
                    'error_type': 'prediction_error'
                }
            }
            return 0.0, detail
        
        # Evaluate response based on metric
        score, metadata = self._compute_score(
            response=response,
            item=item,
            audio_input_path=audio_input_path,
            audio_reference_path=audio_reference_path,
            latency=latency,
        )
        
        # Prepare detail dictionary
        detail = {
            'item_id': item_id,
            'task': item['task'],
            'dataset': item['dataset'],
            'metric': item['metric'],
            'score': score,
            'audio_output_expected': item['audio_output'],
            'text_output_expected': item['text_output'],
            'latency': latency,
            'metadata': metadata
        }
        
        return score, detail


    
    def _eval_function_calling_item(
        self,
        item: Dict[str, Any],
        initial_message: Message,
        predict_fn: Callable,
        audio_input_path: Optional[str],
        audio_reference_path: Optional[str]
    ) -> tuple:
        """
        Evaluate a function calling item with multi-turn conversation
        
        Args:
            item: Benchmark item from dataset
            initial_message: Initial user message
            predict_fn: Model prediction function
            audio_input_path: Path to audio input file (if applicable)
            audio_reference_path: Path to audio reference file (if applicable)
            
        Returns:
            Tuple of (score, detail_dict)
        """
        item_id = item['item_id']
        
        # Load functions from cava/functions.json
        tools = None
        try:
            # Construct path to functions.json relative to current file
            function_file = Path(__file__).parent / "cava" / "functions.json"
            
            if function_file.exists():
                with open(function_file, "r") as f:
                    tools = json.load(f)
            else:
                print(f"Warning: functions.json not found at {function_file}")
        except Exception as e:
            print(f"Error loading functions.json: {e}")
        
        # Initialize conversation history and tracking
        conversation_messages = [initial_message]
        all_function_calls = []
        final_response_text = ""
        total_latency = 0.0
        max_iterations = 10
        
        try:
            # Initial API call with tools
            start_time = time.time()
            response = predict_fn(
                conversation_messages,
                audio_output=item['audio_output'],
                text_output=item['text_output'],
                tools=tools,
                tool_choice="required"
            )
            end_time = time.time()
            total_latency += (end_time - start_time)
            
            # Update final response text
            if response.text:
                final_response_text = response.text
            
            # Multi-turn conversation loop for function calling
            iteration = 0
            while (
                response.tool_calls is not None
                and len(response.tool_calls) > 0
                and iteration < max_iterations
            ):
                iteration += 1
                
                # Add assistant message with tool calls to conversation
                conversation_messages.append(Message(
                    role="assistant",
                    text_input=response.text,
                    tool_calls=response.tool_calls
                ))
                
                # Process each tool call
                for tool_call in response.tool_calls:
                    # Extract function call information
                    # Handle both formats:
                    # 1. Standard: {"id": "...", "type": "function", "function": {"name": ..., "arguments": ...}}
                    # 2. Simplified: {"name": ..., "arguments": ..., "id": "..."}
                    
                    if 'function' in tool_call:
                        # Standard format with nested function object
                        func_data = tool_call['function']
                        func_name = func_data['name']
                        func_args = func_data.get('arguments', {})
                        func_id = tool_call.get('id')
                    elif 'name' in tool_call:
                        # Simplified format with direct name and arguments
                        func_name = tool_call['name']
                        func_args = tool_call.get('arguments', {})
                        func_id = tool_call.get('id')
                    else:
                        # Unknown format, skip
                        continue
                    
                    # Store function call in standardized format
                    function_call = {
                        "name": func_name,
                        "arguments": func_args
                    }
                    all_function_calls.append(function_call)
                    
                    # Add mock function result to conversation
                    # tool_call_id is optional - only needed for OpenAI-like APIs
                    # Models like Gemini don't require it for tracking
                    mock_result = f"MOCK_RESPONSE({func_name})"
                    tool_message = Message(
                        role="tool",
                        text_input=mock_result,
                        name=func_name  # Use function name for identification
                    )
                    # Only add tool_call_id if it exists (for OpenAI compatibility)
                    if func_id:
                        tool_message.tool_call_id = func_id
                    conversation_messages.append(tool_message)
                
                # Continue conversation with another API call
                start_time = time.time()
                response = predict_fn(
                    conversation_messages,
                    audio_output=item['audio_output'],
                    text_output=item['text_output'],
                    tools=tools,
                    tool_choice="auto"
                )
                end_time = time.time()
                total_latency += (end_time - start_time)
                
                # Update final response text
                if response.text:
                    final_response_text = response.text
            
            # Evaluate the function calls using the metric directly
            text_reference = item.get('text_reference')
            if text_reference and isinstance(text_reference, list) and len(text_reference) > 0:
                text_reference = text_reference[0]
            
            score, metadata = function_calling_match(
                model_response_tool_calls=all_function_calls if all_function_calls else None,
                text_reference=text_reference
            )
            
            # Add latency and conversation metadata
            metadata['latency'] = total_latency
            metadata['function_calls'] = all_function_calls
            metadata['num_iterations'] = iteration
            metadata['num_function_calls'] = len(all_function_calls)
            
            # Prepare detail dictionary
            detail = {
                'item_id': item_id,
                'task': item['task'],
                'dataset': item['dataset'],
                'metric': item['metric'],
                'score': score,
                'audio_output_expected': item['audio_output'],
                'text_output_expected': item['text_output'],
                'latency': total_latency,
                'metadata': metadata
            }
            
            return score, detail
            
        except Exception as e:
            # Function calling evaluation failed
            detail = {
                'item_id': item_id,
                'task': item['task'],
                'dataset': item['dataset'],
                'metric': item['metric'],
                'score': 0.0,
                'audio_output_expected': item['audio_output'],
                'text_output_expected': item['text_output'],
                'latency': total_latency,
                'function_calls': all_function_calls,
                'metadata': {
                    'error': f"Function calling evaluation failed: {str(e)}",
                    'error_type': 'function_calling_error',
                    'latency': total_latency
                }
            }
            return 0.0, detail
        
    def _compute_score(
        self,
        response: ModelResponse,
        item: Dict[str, Any],
        audio_input_path: Optional[str],
        audio_reference_path: Optional[str],
        latency: Optional[float] = None,
    ) -> tuple[float, Dict[str, Any]]:
        """
        Compute score for a response based on the metric type
        
        Args:
            response: Model response
            item: Benchmark item
            audio_input_path: Path to audio input file (if applicable)
            audio_reference_path: Path to audio reference file (if applicable)
            latency: Response latency in seconds
            
        Returns:
            Tuple of (score, metadata_dict)
            - score: float in [0, 1]
            - metadata: dict with evaluation details
        """
        dataset = item['dataset']
        task = item['task']
        metric = item['metric']
        
        metadata = {
            'dataset': dataset,
            'task': task,
            'metric': metric,
            'latency': latency,
        }
        
        try:
            if dataset == "dynamic_superb":
                # Dynamic SuperB expects text output
                if response.text is None or response.text.strip() == '':
                    return 0.0, {
                        'error': 'Response text is None or empty',
                        'error_type': 'missing_text_output',
                        'reference': item['text_reference'][0],
                        'latency': latency,
                    }
                    
                if metric == 'llm_classification':
                    score, eval_metadata = llm_classification(
                        response.text, 
                        item['text_reference'][0],
                        item['text_input']
                    )
                    metadata.update(eval_metadata)
                    metadata['reference'] = item['text_reference'][0]
                    return score, metadata
                    
                elif metric == 'phoneme_error_rate':
                    score, eval_metadata = phoneme_error_rate(
                        model_response_text=response.text,
                        text_reference=item['text_reference'][0]
                    )
                    metadata.update(eval_metadata)
                    return score, metadata

                elif metric == 'word_error_rate':
                    score, eval_metadata = word_error_rate(
                        model_response_text=response.text,
                        text_reference=item['text_reference'][0]
                    )
                    metadata.update(eval_metadata)
                    return score, metadata

                elif metric == 'pos_estimation':
                    score, eval_metadata = pos_estimation(
                        model_response_text=response.text,
                        text_reference=item['text_reference'][0],
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                                
                else:
                    metadata['error'] = f'Unknown metric for dynamic_superb: {metric}'
                    metadata['error_type'] = 'unknown_metric'
                    return 0.0, metadata
            
            elif dataset == "ultraeval":
                # UltraEval expects audio output
                if response.audio_path is None or not os.path.exists(response.audio_path):
                    return 0.0, {
                        'error': 'Response audio_path is None or does not exist',
                        'error_type': 'missing_audio_output',
                        'reference': item.get('text_reference'),
                        'latency': latency,
                    }
                
                if metric == 'GPT-score':
                    metadata_json = item.get('metadata')
                    
                    score, eval_metadata = ultraeval_gpt_score(
                        model_response_audio_path=response.audio_path,
                        metadata_json=metadata_json,
                        transcriber=self.whisper_transcriber,
                        model_name="gpt-4o-mini",
                        max_retries=3
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                
                elif metric == 'ExistMatch':
                    
                    score, eval_metadata = ultraeval_exist_match(
                        model_response_audio_path=response.audio_path,
                        text_reference=item['text_reference'],
                        transcriber=self.whisper_transcriber
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                
                else:
                    metadata['error'] = f'Unknown metric for ultraeval: {metric}'
                    metadata['error_type'] = 'unknown_metric'
                    return 0.0, metadata
            elif dataset == "wildspeech-bench":
                # WildSpeech-Bench expects audio output
                if response.audio_path is None or not os.path.exists(response.audio_path):
                    return 0.0, {
                        'error': 'Response audio_path is None or does not exist',
                        'error_type': 'missing_audio_output',
                        'reference': item.get('text_reference'),
                        'latency': latency,
                    }
                
                if metric == 'GPT-Score':
                    metadata_json = item.get('metadata')
                    text_reference = item.get('text_reference')
                    if isinstance(text_reference, list) and len(text_reference) > 0:
                        text_reference = text_reference[0]
                    elif text_reference is None:
                        text_reference = ""
                    
                    score, eval_metadata = wildspeech_gpt_score(
                        model_response_audio_path=response.audio_path,
                        text_reference=text_reference,
                        metadata_json=metadata_json,
                        transcriber=self.whisper_transcriber,
                        model_name="gpt-4o-mini"
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                
                elif metric == 'UTMOS':
                    score, eval_metadata = wildspeech_utmos(
                        model_response_audio_path=response.audio_path
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                
                else:
                    metadata['error'] = f'Unknown metric for wildspeech-bench: {metric}'
                    metadata['error_type'] = 'unknown_metric'
                    return 0.0, metadata
            elif dataset == "speakbench":
                # SpeakBench expects audio output
                if response.audio_path is None or not os.path.exists(response.audio_path):
                    return 0.0, {
                        'error': 'Response audio_path is None or does not exist',
                        'error_type': 'missing_audio_output',
                        'audio_reference': audio_reference_path,
                        'latency': latency,
                    }
                
                    
                score, eval_metadata = speakbench_WinRate(
                    model_response_audio_path=response.audio_path,
                    audio_input_path=audio_input_path,
                    audio_reference_path=audio_reference_path,
                    concat_test=True
                )
                metadata.update(eval_metadata)
                return score, metadata
            elif dataset == "cava":
                if task == "emotion":
                    if response.text is None or response.text.strip() == '':
                        return 0.0, {
                            'error': 'Response text is None or empty',
                            'error_type': 'missing_text_output',
                            'text_reference': item['text_reference'][0],
                            'latency': latency,
                        }
                    score, eval_metadata = emotion_match(
                        response.text.strip(), 
                        item['text_reference'][0]
                    )
                    metadata.update(eval_metadata)
                    metadata['text_reference'] = item['text_reference'][0]
                    return score, metadata
                if task == "speaker_diarization":
                    if response.text is None or response.text.strip() == '':
                        return 0.0, {
                            'error': 'Response text is None or empty',
                            'error_type': 'missing_text_output',
                            'text_reference': item['text_reference'][0],
                            'latency': latency,
                        }
                    score, eval_metadata = speaker_diarization_jer(
                        response.text, 
                        item['text_reference']
                    )
                    metadata.update(eval_metadata)
                    metadata['text_reference'] = item['text_reference']
                    return score, metadata
                if task == "multimodal_instruction_following":
                    if response.text is None or response.text.strip() == '':
                        return 0.0, {
                            'error': 'Response text is None or empty',
                            'error_type': 'missing_text_output',
                            'text_reference': item['text_reference'][0],
                            'latency': latency,
                        }
                    score, eval_metadata = multimodal_instruction_following(
                        response.text.strip(), 
                        item['metadata']
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                if task == "deception_detection":
                    if response.text is None or response.text.strip() == '':
                        return 0.0, {
                            'error': 'Response text is None or empty',
                            'error_type': 'missing_text_output',
                            'text_reference': item['text_reference'][0],
                            'latency': latency,
                        }
                    score, eval_metadata = deception_detection_exact_match(
                        response.text.strip(), 
                        item['text_reference']
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                if task == "jailbreak" or task == "jailbreak_base":
                    if response.text is None or response.text.strip() == '':
                        return 0.0, {
                            'error': 'Response text is None or empty',
                            'error_type': 'missing_text_output',
                            'latency': latency,
                        }
                    score, eval_metadata = jailbreak_refusal_rate(
                        response.text.strip()
                    )
                    metadata.update(eval_metadata)
                    return score, metadata

                if task == "jeopardy_correctness":
                    if response.text is None or response.text.strip() == '':
                        return 0.0, {
                            'error': 'Response text is None or empty',
                            'error_type': 'missing_text_output',
                            'text_reference': item['text_reference'][0],
                            'latency': latency,
                        }
                    
                    score, eval_metadata = jeopardy_correctness_pedant(
                        model_response_text=response.text.strip(),
                        text_reference=item['text_reference'][0],
                        metadata_json=item['metadata']
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                
                if task == "jeopardy_latency":
                    if response.text is None or response.text.strip() == '':
                        return 0.0, {
                            'error': 'Response text is None or empty',
                            'error_type': 'missing_text_output',
                            'text_reference': item['text_reference'][0],
                            'latency': latency,
                        }
                    
                    score, eval_metadata = jeopardy_latency_score(
                        model_response_text=response.text.strip(),
                        text_reference=item['text_reference'][0],
                        metadata_json=item['metadata'],
                        latency=latency
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                if task == "pronunciation_oed" or task == "pronunciation_audio":
                    # Pronunciation tasks expect audio output
                    if response.audio_path is None or not os.path.exists(response.audio_path):
                        return 0.0, {
                            'error': 'Response audio_path is None or does not exist',
                            'error_type': 'missing_audio_output',
                            'audio_reference': audio_reference_path,
                            'latency': latency,
                        }
                    
                    score, eval_metadata = pronunciation_match(
                        model_response_audio_path=response.audio_path,
                        audio_reference_path=audio_reference_path,
                    )
                    metadata.update(eval_metadata)
                    return score, metadata
                else:
                    metadata['error'] = f'Unknown metric for cava: {metric}'
                    metadata['error_type'] = 'unknown_metric'
                    return 0.0, metadata
            else:
                metadata['error'] = f'Unknown dataset: {dataset}'
                metadata['error_type'] = 'unknown_dataset'
                return 0.0, metadata
                
        except Exception as e:
            metadata['error'] = f'Evaluation failed: {str(e)}'
            metadata['error_type'] = 'evaluation_error'
            if item.get('text_reference'):
                metadata['text_reference'] = item.get('text_reference')
            if item.get('audio_reference'):
                metadata['audio_reference'] = audio_reference_path
            return 0.0, metadata