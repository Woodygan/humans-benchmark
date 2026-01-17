import re
from typing import Dict, Any, Tuple
from openai import OpenAI
import os
import string
from speech_metrics import per, wer


def create_classification_prompt(instruction: str, label: str, sllm_response: str) -> list:
    """Create prompt for LLM classification evaluation"""
    INSTRUCTION_REFINE_COT_SINGLE_AND_MULTI_CHOICE = """
You will be given a question, a corresponding correct answer(s), and a response from a model.
The model's response is a reply to the question. Your task is to judge if the "Model's Response" aligns with the "Ground Truth Answer" based on the "Question."
Please strictly follow the guidelines below:
- Briefly explain the reasons for your judgment.
- Answer with the format "Result: <YES or NO>" at the end.
- Output "YES" if the response aligns with the ground truth answer; 
output "NO" if the response does not match the ground truth answer, selects incorrect or irrelevant options, or provides more answers than required.
- The questions would be single-choice or multi-choice:
For single-choice questions, the model's response should contain one and only one answer. If the model's response selects more than one answer or does not clearly indicate a single answer, you should mark it as incorrect and output "NO." 
For multi-choice questions, the model's response must exactly match all applicable correct choices. If the model's response selects too many, too few, or any incorrect answers, you should mark it as incorrect and output "NO."
- Since the question is short answer, the model's response does not need to mention the content of the question. You only need to check if the model's response has the same meaning as the ground truth answer(s).

Input Format:
Question: {instruction}
Ground Truth Answer: {label}
Model's Response: {sllm_response}
""".strip()

    text = INSTRUCTION_REFINE_COT_SINGLE_AND_MULTI_CHOICE.format(
        instruction=instruction, 
        label=label, 
        sllm_response=sllm_response
    )

    messages = [
        {"role": "user", "content": text},
    ]
    
    return messages


def read_result(text: str) -> str:
    """Extract yes/no result from LLM response"""
    result = re.search(r"result:\s*(yes|no)", text.lower().strip())
    return result.group(1) if result else None


def llm_classification(
    model_response_text: str, 
    text_reference: str,
    instruction: str = "",
    model_name: str = "gpt-4o",
    max_retries: int = 3
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate model response using LLM classification
    
    Args:
        model_response_text: The model's text output
        text_reference: Ground truth answer
        instruction: Optional instruction/question text
        model_name: OpenAI model to use for evaluation
        max_retries: Number of retry attempts
        
    Returns:
        Tuple of (score, metadata)
        - score: 1.0 if correct, 0.0 if incorrect
        - metadata: dict with evaluation details
    """
    import time
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create prompt
    messages = create_classification_prompt(
        instruction=instruction,
        label=text_reference,
        sllm_response=model_response_text
    )
    
    # Generate LLM evaluation with retry logic
    llm_response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=512,
                temperature=0.0,
            )
            
            if response.choices and response.choices[0].message:
                llm_response = response.choices[0].message.content
                if llm_response:
                    llm_response = llm_response.strip()
                    break
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 1.0 * (2 ** attempt)  # Exponential backoff
                print(f"LLM evaluation error, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                print(f"LLM evaluation failed after {max_retries} attempts: {str(e)}")
                # Return 0 score with error metadata
                return 0.0, {
                    'text_answer': model_response_text,
                    'text_ref': text_reference,
                    'llm_response': None,
                    'error': f"LLM evaluation failed: {str(e)}",
                    'error_type': 'llm_evaluation_error'
                }
    
    # Parse result from LLM response
    if llm_response:
        result = read_result(llm_response)
        score = 1.0 if result == "yes" else 0.0
    else:
        # No response received
        result = None
        score = 0.0
    
    # Prepare metadata
    metadata = {
        'text_answer': model_response_text,
        'text_ref': text_reference,
        'llm_response': llm_response,
        'parsed_result': result,
    }
    
    # Add error if result couldn't be parsed
    if result is None and llm_response:
        metadata['error'] = 'Could not parse yes/no from LLM response'
        metadata['error_type'] = 'parse_error'
    
    return score, metadata


def normalization(text: str) -> str:
    """Normalize text by removing punctuation and converting to lowercase"""
    return text.translate(str.maketrans("", "", string.punctuation)).lower()


def phoneme_error_rate(
    model_response_text: str, 
    text_reference: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute phoneme error rate between model response and reference
    
    Args:
        model_response_text: The model's text output
        text_reference: Ground truth text
        
    Returns:
        Tuple of (score, metadata)
        - score: 1 - PER (bounded to [0, 1], higher is better)
        - metadata: dict with evaluation details
    """
    # Normalize both texts
    normalized_response = normalization(model_response_text)
    normalized_reference = normalization(text_reference)
    
    try:
        # Compute phoneme error rate (single example)
        # per() expects lists, so wrap in lists
        error_rate = per([normalized_response], [normalized_reference])
        
        # Bound the score to [0, 1] range
        # PER can theoretically be > 1.0 if there are many insertions
        # Score = 1 - PER, where higher score is better
        score = max(0.0, 1.0 - error_rate)
        
        # Prepare metadata
        metadata = {
            'text_answer': model_response_text,
            'text_answer_normalized': normalized_response,
            'text_ref': text_reference,
            'text_ref_normalized': normalized_reference,
            'phoneme_error_rate': error_rate,
        }
        
        return score, metadata
        
    except Exception as e:
        # Handle errors in PER calculation
        print(f"Error computing phoneme error rate: {e}")
        
        metadata = {
            'text_answer': model_response_text,
            'text_ref': text_reference,
            'error': f"PER computation failed: {str(e)}",
            'error_type': 'per_computation_error'
        }
        
        return 0.0, metadata


def normalization(text: str) -> str:
    """Normalize text by removing punctuation and converting to lowercase"""
    return text.translate(str.maketrans("", "", string.punctuation)).lower()


def word_error_rate(
    model_response_text: str, 
    text_reference: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute word error rate between model response and reference
    
    Args:
        model_response_text: The model's text output
        text_reference: Ground truth text
        
    Returns:
        Tuple of (score, metadata)
        - score: 1 - WER (bounded to [0, 1], higher is better)
        - metadata: dict with evaluation details
    """
    # Normalize both texts
    normalized_response = normalization(model_response_text)
    normalized_reference = normalization(text_reference)
    
    try:
        # Compute word error rate (single example)
        # wer() expects lists, so wrap in lists
        error_rate = wer([normalized_response], [normalized_reference])
        
        # Bound the score to [0, 1] range
        # WER can theoretically be > 1.0 if there are many insertions
        # Score = 1 - WER, where higher score is better
        score = max(0.0, 1.0 - error_rate)
        
        # Prepare metadata
        metadata = {
            'text_answer': model_response_text,
            'text_answer_normalized': normalized_response,
            'text_ref': text_reference,
            'text_ref_normalized': normalized_reference,
            'word_error_rate': error_rate,
        }
        
        return score, metadata
        
    except Exception as e:
        # Handle errors in WER calculation
        print(f"Error computing word error rate: {e}")
        
        metadata = {
            'text_answer': model_response_text,
            'text_ref': text_reference,
            'error': f"WER computation failed: {str(e)}",
            'error_type': 'wer_computation_error'
        }
        
        return 0.0, metadata


def pos_estimation(
    model_response_text: str, 
    text_reference: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute Part of Speech (PoS) estimation score using WER
    
    This function is designed for Part of Speech (PoS) estimation.
    It calculates the Word Error Rate (WER) between the hypothesized sentence 
    and the reference sentence.
    
    Args:
        model_response_text: The model's text output
        text_reference: Ground truth text
        
    Returns:
        Tuple of (score, metadata)
        - score: 1 - WER (bounded to [0, 1], higher is better)
        - metadata: dict with evaluation details
    """
    
    def preprocess_text(text: str) -> str:
        """Preprocess text for PoS estimation"""
        
        # Remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)
    
    # Preprocess both texts
    processed_response = preprocess_text(model_response_text)
    processed_reference = preprocess_text(text_reference)
    
    try:
        # Compute word error rate (single example)
        # wer() expects lists, so wrap in lists
        error_rate = wer([processed_response], [processed_reference])
        
        # Bound the score to [0, 1] range
        # Score = 1 - WER, where higher score is better
        score = max(0.0, 1.0 - error_rate)
        
        # Prepare metadata
        metadata = {
            'text_answer': model_response_text,
            'text_answer_processed': processed_response,
            'text_ref': text_reference,
            'text_ref_processed': processed_reference,
            'word_error_rate': error_rate,
        }
        
        return score, metadata
        
    except Exception as e:
        # Handle errors in WER calculation
        print(f"Error computing PoS estimation: {e}")
        
        metadata = {
            'text_answer': model_response_text,
            'text_ref': text_reference,
            'error': f"PoS estimation failed: {str(e)}",
            'error_type': 'pos_estimation_error'
        }
        
        return 0.0, metadata