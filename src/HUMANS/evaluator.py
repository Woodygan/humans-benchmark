from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal, Callable
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

class HUMANSEvaluator:
    """
    HUMANS Benchmark evaluator
    
    Efficiently evaluates Large Audio Models using minimal subsets while
    predicting human preferences through learned regression weights.
    """
    
    def __init__(
        self,
        dataset_name: str = "EfficientAudioBench/HUMANS",
        subset: str = "n50",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize HUMANS Benchmark
        
        Args:
            dataset_name: HuggingFace dataset name
            subset: Subset to use (n10, n20, n30, n50, n100, n200)
            cache_dir: Cache directory for dataset
        """
        print(f"Loading HUMANS Benchmark: {dataset_name} ({subset})...")
        self.dataset_name = dataset_name
        self.subset = subset
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset(dataset_name, subset, cache_dir=cache_dir)
        self.items = list(self.dataset[subset])
        
        print(f"✓ Loaded {len(self.items)} evaluation items")
        
        # Extract weights and bias
        self.human_weights = np.array([item['human_preference_weight'] for item in self.items])
        self.benchmark_weights = np.array([item['full_benchmark_weight'] for item in self.items])
        self.regression_bias = self.items[0]['human_regression_bias']
        
        print(f"✓ Ready to evaluate (regression bias: {self.regression_bias:.4f})")
    
    def evaluate(
        self,
        predict_fn: Callable,
        mode: str = "regression",
        return_details: bool = False,
        max_turns: int = 10,
        tool_executor: Optional[Callable] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on HUMANS Benchmark
        
        Args:
            predict_fn: Function (messages, tools, tool_choice) -> ModelResponse
                       Takes conversation messages and returns model response
            mode: "regression" for human preference prediction (default)
                  "benchmark" for full benchmark score approximation
            return_details: If True, return per-item results and task breakdown
            max_turns: Maximum conversation turns for function calling tasks
            tool_executor: Optional function (func_name, args) -> result
                          Defaults to mock executor if not provided
            verbose: Show progress bar during evaluation
            
        Returns:
            Dictionary with evaluation results:
            - human_preference_score (if mode="regression"): Predicted user satisfaction
            - benchmark_score (if mode="benchmark"): Approximate full benchmark score
            - mean_item_score: Average score across all items
            - num_items: Number of items evaluated
            - details (if return_details=True): Per-item results
            - task_breakdown (if return_details=True): Scores by task
        """
        # Default mock tool executor
        if tool_executor is None:
            tool_executor = lambda name, args: f"MOCK_RESULT({name})"
        
        scores = []
        details = []
        
        iterator = tqdm(self.items, desc="Evaluating") if verbose else self.items
        
        for item in iterator:
            try:
                score, detail = self._eval_item(item, predict_fn, tool_executor, max_turns)
                scores.append(score)
                if return_details:
                    details.append(detail)
            except Exception as e:
                if verbose:
                    print(f"\n⚠ Error on item {item['item_id']}: {e}")
                scores.append(0.0)
                if return_details:
                    details.append({
                        'item_id': item['item_id'],
                        'error': str(e),
                        'score': 0.0,
                    })
        
        scores = np.array(scores)
        
        # Compute final score using weights
        if mode == "regression":
            final_score = np.dot(self.human_weights, scores) + self.regression_bias
            score_key = "human_preference_score"
        else:  # mode == "benchmark"
            final_score = np.dot(self.benchmark_weights, scores)
            score_key = "benchmark_score"
        
        results = {
            score_key: float(final_score),
            "mean_item_score": float(np.mean(scores)),
            "std_item_score": float(np.std(scores)),
            "num_items": len(scores),
            "subset": self.subset,
        }
        
        if return_details:
            results["details"] = details
            results["task_breakdown"] = self._compute_task_breakdown(details)
        
        return results
    
    def _eval_item(self, item, predict_fn, tool_executor, max_turns):
        """Evaluate a single benchmark item"""
        from HUMANS.metrics import compute_metric
        
        # Build conversation messages
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(
                role="user",
                content=item['prompt'],
                audio_path=item['audio_input']['path'] if item['audio_input'] else None
            )
        ]
        
        # Parse metadata for task-specific information
        metadata = json.loads(item['metadata']) if item['metadata'] else {}
        tools = metadata.get('tools')
        tool_choice = metadata.get('tool_choice', 'auto')
        
        # Get model response
        if tools:
            # Function calling task - multi-turn conversation
            response, tool_calls = self._run_function_calling(
                messages, tools, tool_choice, predict_fn, tool_executor, max_turns
            )
        else:
            # Standard task - single turn
            response = predict_fn(messages, None, None)
            tool_calls = None
        
        # Compute score using task-specific metric
        score = compute_metric(
            metric=item['metric'],
            prediction=response.text,
            reference=item['text_reference'],
            prediction_audio=response.audio_path,
            reference_audio=item['audio_reference']['path'] if item['audio_reference'] else None,
            tool_calls=tool_calls,
            expected_calls=metadata.get('expected_function_calls'),
        )
        
        # Prepare detailed results
        detail = {
            'item_id': item['item_id'],
            'task': item['task'],
            'dataset': item['dataset'],
            'score': score,
            'prediction': response.text,
            'metric': item['metric'],
        }
        
        if response.audio_path:
            detail['prediction_audio'] = response.audio_path
        
        if tool_calls:
            detail['tool_calls'] = tool_calls
        
        return score, detail
    
    def _run_function_calling(
        self, messages, tools, tool_choice, predict_fn, tool_executor, max_turns
    ):
        """Handle multi-turn function calling conversation"""
        all_tool_calls = []
        final_response = None
        
        for turn in range(max_turns):
            # Get model response
            response = predict_fn(messages, tools, tool_choice)
            
            # Check if model made function calls
            if not response.tool_calls:
                # No more tool calls - this is the final response
                final_response = response
                break
            
            # Add assistant message with tool calls
            messages.append(Message(
                role="assistant",
                content=response.text,
                tool_calls=response.tool_calls
            ))
            
            # Execute each tool call and add results
            for tool_call in response.tool_calls:
                func_name = tool_call['function']['name']
                func_args = json.loads(tool_call['function']['arguments'])
                
                # Track the call
                all_tool_calls.append({
                    'name': func_name,
                    'arguments': func_args
                })
                
                # Execute tool
                result = tool_executor(func_name, func_args)
                
                # Add tool response to conversation
                messages.append(Message(
                    role="tool",
                    tool_call_id=tool_call['id'],
                    name=func_name,
                    content=result
                ))
            
            # After first turn, don't force tool calls
            tool_choice = 'auto'
        
        # If we hit max turns without a final response, use last response
        if final_response is None:
            final_response = response
        
        return final_response, all_tool_calls
    
    def _compute_task_breakdown(self, details):
        """Compute average scores grouped by task"""
        from collections import defaultdict
        
        task_scores = defaultdict(list)
        for detail in details:
            if 'score' in detail and 'task' in detail:
                task_scores[detail['task']].append(detail['score'])
        
        return {
            task: float(np.mean(scores))
            for task, scores in task_scores.items()
        }