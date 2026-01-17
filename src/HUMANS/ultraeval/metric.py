import re
import json
import os
import unicodedata
from typing import Dict, Any, Tuple, Optional
import regex
from openai import OpenAI
import time
from ..whisper import WhisperTranscriber, whisper_transcribe

# =============================================================================
# GPT-SCORE METRIC (UltraEval)
# =============================================================================

def ultraeval_gpt_score(
    model_response_audio_path: str,
    metadata_json: str,
    transcriber: Optional[WhisperTranscriber] = None,
    model_name: str = "gpt-4o-mini",
    max_retries: int = 3
) -> Tuple[float, Dict[str, Any]]:
    """
    UltraEval GPT-Score evaluation (exactly following paste 2 logic)
    
    Pipeline: Audio → Whisper Transcription → GPT-4 Judge
    
    Args:
        model_response_audio_path: Path to model's audio output
        metadata_json: JSON string containing instruction
        reference_text: Reference answer/output
        transcriber: WhisperTranscriber instance
        model_name: GPT model to use as judge (default: gpt-4o-mini)
        max_retries: Number of retry attempts
        
    Returns:
        Tuple of (score, metadata)
        - score: normalized to [0, 1] using (raw_score - 1) / 9
        - metadata: dict with evaluation details
    """
    metadata_dict = json.loads(metadata_json)
    instruction = metadata_dict.get('instruction')
    # Step 1: Transcribe audio using Whisper
    pred_text, whisper_metadata = whisper_transcribe(model_response_audio_path, transcriber)
    
    if 'error' in whisper_metadata:
        # Transcription failed
        return 0.0, whisper_metadata
    
    # Step 2: Build GPT-4 judge prompt (exactly as in paste 2)
    CHATBOT_EVAL_PROMPT = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided
by an AI assistant to the user question displayed below. Your evaluation should
consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of the response. Begin your evaluation by providing a short
explanation. Be as objective as possible. After providing your explanation, you
must rate the response on a scale of 1 to 10 by strictly following this format:
"[[rating]]", for example: "Rating: [[5]]".

[Question]
{instruction}

[The Start of Assistant's Answer]
{response}
[The End of Assistant's Answer]"""

    prompt = CHATBOT_EVAL_PROMPT.format(
        instruction=instruction,
        response=pred_text
    )
    
    # Step 3: Call GPT-4 as judge
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    gpt_response = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=2048
            )
            
            gpt_response = response.choices[0].message.content
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 1.0 * (2 ** attempt)  # Exponential backoff
                print(f"GPT-4 evaluation error, retrying in {wait_time}s: {str(e)}")
                time.sleep(wait_time)
            else:
                print(f"GPT-4 evaluation failed after {max_retries} attempts: {str(e)}")
                return 0.0, {
                    'text_answer': pred_text,
                    'error': f"GPT-4 evaluation failed: {str(e)}",
                    'error_type': 'gpt_evaluation_error'
                }
    
    # Step 4: Parse rating [[X]]
    rating = 5  # Default middle score
    if gpt_response:
        match = re.search(r'\[\[(\d+)\]\]', gpt_response)
        if match:
            rating = int(match.group(1))
        else:
            print(f"Warning: Could not parse rating from GPT response: {gpt_response}")
    
    # Step 5: Normalize score to [0, 1] using (rating - 1) / 9
    # Rating is 1-10, so (1-1)/9 = 0 and (10-1)/9 = 1
    normalized_score = min(max(0.0,(rating - 1) / 9.0),1.0)
    
    # Prepare metadata
    metadata = {
        'text_answer': pred_text,
        'instruction': instruction,
        'gpt_response': gpt_response,
        'raw_rating': rating,
        'normalized_score': normalized_score,
    }
    
    return normalized_score, metadata


# =============================================================================
# EXIST MATCH METRIC (UltraEval)
# =============================================================================

class SimpleTokenizer:
    """Tokenizer from UltraEval QA evaluation code"""
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def ultraeval_exist_match(
    model_response_audio_path: str,
    text_reference: Any,  # Can be string or list of strings
    transcriber: Optional[WhisperTranscriber] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    UltraEval ExistMatch evaluation (exactly following paste 2 logic)
    
    Pipeline: Audio → Whisper Transcription → Token-level Substring Match
    
    Checks if ANY reference answer exists as substring in prediction (token-level).
    Exactly implements QAExistMatchEvaluator from paste 2.
    
    Args:
        model_response_audio_path: Path to model's audio output
        text_reference: Reference answer(s) - string or list of strings
        transcriber: WhisperTranscriber instance
        
    Returns:
        Tuple of (score, metadata)
        - score: 1.0 if match found, 0.0 otherwise
        - metadata: dict with evaluation details
    """
    from typing import List, Any
    
    # Step 1: Transcribe audio using Whisper
    pred_text, whisper_metadata = whisper_transcribe(model_response_audio_path, transcriber)
    
    if 'error' in whisper_metadata:
        # Transcription failed
        return 0.0, whisper_metadata
    
    # Step 2: Convert reference to list if needed (exactly as in paste 2)
    if isinstance(text_reference, str):
        answers = [text_reference]
    else:
        answers = text_reference
    
    # Step 3: Check if any answer exists in prediction (token-level)
    # Exactly implements has_answer() from QAExistMatchEvaluator in paste 2
    tokenizer = SimpleTokenizer()
    
    # Normalize and tokenize prediction text (NFD normalization as in paste 2)
    pred_normalized = unicodedata.normalize('NFD', pred_text)
    pred_tokens = tokenizer.tokenize(pred_normalized, uncased=True)
    
    # Check each answer
    match_found = False
    matched_answer = None
    
    for answer in answers:
        # Normalize and tokenize answer
        answer_normalized = unicodedata.normalize('NFD', answer)
        answer_tokens = tokenizer.tokenize(answer_normalized, uncased=True)
        
        # Check if answer tokens appear as contiguous substring in prediction
        for i in range(0, len(pred_tokens) - len(answer_tokens) + 1):
            if answer_tokens == pred_tokens[i: i + len(answer_tokens)]:
                match_found = True
                matched_answer = answer
                break
        
        if match_found:
            break
    
    score = 1.0 if match_found else 0.0
    
    # Prepare metadata (following paste 2's evaluate() return format)
    metadata = {
        'match': 1 if match_found else 0,
        'pred': matched_answer if match_found else pred_text,  # Original behavior from paste 2
        'ref': answers,
        'text_answer': pred_text,
        'text_answer_normalized': pred_normalized,
        'text_answer_tokens': pred_tokens,
    }
    
    return score, metadata