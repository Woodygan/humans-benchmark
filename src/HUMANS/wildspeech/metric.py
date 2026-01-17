import re
import json
import os
import time
from typing import Dict, Any, Tuple, Optional, List
from openai import OpenAI

# Import from whisper module
from ..whisper import WhisperTranscriber


# =============================================================================
# HALLUCINATION DETECTION
# =============================================================================

def has_consecutive_repeated_sentence(text: str, min_repeat: int = 3, min_len: int = 4) -> bool:
    """
    Detect Whisper hallucinations by checking for repeated sentences
    
    Checks if the transcription has the same sentence repeated 3+ times in a row,
    which indicates hallucination/error.
    
    Args:
        text: Transcribed text
        min_repeat: Minimum number of consecutive repeats to flag (default: 3)
        min_len: Minimum sentence length to consider (default: 4)
        
    Returns:
        True if hallucination detected, False otherwise
    """
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) >= min_len]
    count = 1
    for i in range(1, len(sentences)):
        if sentences[i] == sentences[i-1]:
            count += 1
            if count >= min_repeat:
                return True
        else:
            count = 1
    return False


# =============================================================================
# ROBUST WHISPER TRANSCRIPTION (Exact WildSpeech-Bench Logic)
# =============================================================================

def wildspeech_transcribe_audio(
    audio_path: str,
    transcriber: WhisperTranscriber,
    max_transcribe_times: int = 10,
    target_times: int = 3
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Transcribe audio 3 times using Whisper (exactly as in WildSpeech-Bench)
    
    - Transcribes up to 10 times to get 3 valid transcriptions
    - Filters out hallucinations (repeated sentences)
    - Returns 3 good transcriptions
    
    Args:
        audio_path: Path to audio file
        transcriber: WhisperTranscriber instance
        max_transcribe_times: Maximum attempts (default: 10)
        target_times: Target number of valid transcriptions (default: 3)
        
    Returns:
        Tuple of (transcriptions_list, metadata)
    """
    transcribed_texts = []
    all_transcriptions = []
    
    for i in range(max_transcribe_times):
        try:
            transcribed_text = transcriber.transcribe(audio_path)
            all_transcriptions.append(transcribed_text)
            
            if not has_consecutive_repeated_sentence(transcribed_text):
                transcribed_texts.append(transcribed_text)
            
            if len(transcribed_texts) >= target_times:
                break
                
        except Exception as e:
            print(f"  - Attempt {i+1}: Transcription failed: {e}")
            all_transcriptions.append(None)
    
    # If no valid transcriptions, use first 3 anyway
    if not transcribed_texts and all_transcriptions:
        print("  - Warning: No valid transcriptions, using first 3 anyway")
        transcribed_texts = [t for t in all_transcriptions[:target_times] if t is not None]
    
    if not transcribed_texts:
        return [], {
            'error': f'Failed to get any valid transcriptions after {max_transcribe_times} attempts',
            'error_type': 'transcription_failed',
            'audio_path': audio_path,
            'all_attempts': all_transcriptions
        }
    
    metadata = {
        'audio_path': audio_path,
        'num_valid': len(transcribed_texts),
        'num_attempts': len(all_transcriptions),
        'all_attempts': all_transcriptions
    }
    
    return transcribed_texts, metadata


# =============================================================================
# GPT-SCORE METRIC (Exact WildSpeech-Bench Logic)
# =============================================================================

def generate_text_chat(client: OpenAI, *args, **kwargs):
    """Helper function to call OpenAI API with retries (exactly as in WildSpeech-Bench)"""
    for _ in range(25):
        try:
            response = client.chat.completions.create(*args, **kwargs)
            time.sleep(0.5)
            if response is None:
                time.sleep(5)
                continue
            return response
        except Exception as e:
            print(f"    API error: {e}, retrying...")
            time.sleep(5)
    return None


def extract_rating(llm_output: str) -> Optional[int]:
    """Extract score from GPT evaluation output (exactly as in WildSpeech-Bench)"""
    pattern = r"Score: (?:\[)?(\d+)(?:\])?"
    match = re.search(pattern, llm_output)
    if match:
        return int(match.group(1))
    else:
        return None


def wildspeech_gpt_score(
    model_response_audio_path: str,
    text_reference: str,  # The checklist
    metadata_json: str,   # Contains query (user query)
    transcriber: WhisperTranscriber,
    model_name: str = "gpt-4o-mini"
) -> Tuple[float, Dict[str, Any]]:
    """
    WildSpeech-Bench GPT-Score evaluation (exact reproduction)
    
    Pipeline: 
    1. Audio → Whisper (3 valid transcriptions from max 10 attempts)
    2. For each transcription → GPT-4o-mini judge (3 evaluations)
    3. Average scores → Normalize to [0, 1]
    
    Args:
        model_response_audio_path: Path to model's audio output
        text_reference: Reference checklist for evaluation
        metadata_json: JSON string containing 'query' (user query)
        transcriber: WhisperTranscriber instance
        model_name: GPT model to use (default: gpt-4o-mini)
        
    Returns:
        Tuple of (score, metadata)
        - score: average of GPT scores, normalized to [0, 1] using (avg - 1) / 9
        - metadata: dict with all evaluations and scores
    """
    metadata_dict = json.loads(metadata_json)
    user_query = metadata_dict.get('query')
    
    # Step 1: Transcribe audio (3 valid transcriptions from max 10 attempts)
    transcriptions, whisper_metadata = wildspeech_transcribe_audio(
        audio_path=model_response_audio_path,
        transcriber=transcriber,
        max_transcribe_times=10,
        target_times=3
    )
    
    if 'error' in whisper_metadata:
        return 0.0, whisper_metadata
    
    # Step 2: GPT-4o-mini judges each transcription
    general_checklist = """
- Instruction adherence is a core evaluation metric. If the AI's response does not fully follow the user's querys, it constitutes a serious error.
- Correctness is a core evaluation metric. If the AI's response contains factual inaccuracies, it would be a serious flaw.
- If the AI's response contain a large amount of repetitive content, it would be a serious flaw.
"""

    meta_prompt = """
# Instructions

You are an evaluation expert. Your task is to assess the quality of AI model responses. We will provide you with user queries and AI responses. Please note that both the user queries and AI responses are in audio format. For your convenience, we have converted them into text, but you should evaluate from the perspective of voice communication and analyze the characteristics of voice communication when assessing the quality of the AI response.
You should first carefully read the user query to analyze the task, then evaluate the quality of the response based on the rules provided below.

# Conversation between User and AI

### User Query
<|begin_of_query|>

{query}

<|end_of_query|>

### AI Response
<|begin_of_response|>

{response}

<|end_of_response|>

# Evaluation

## Checklist
<|begin_of_checklist|>

{checklist}

<|end_of_checklist|>

The checklist serves as a guiding framework for your evaluation. However, feel free to consider aspects beyond its contents to ensure a well - rounded assessment.

## Rules

You should evaluate based on the analysis of user questions and AI responses, referring to the contents in the checklist during the evaluation. However, remember that the checklist is meant to provide comprehensive reference information, but it is not the standard answer. Sometimes, the AI response does not need to cover all the contents involved in the checklist to meet user needs, and you need to make this judgment on your own. The scoring scale ranges from 1 to 10:
- 1~2 points: No value/meaningless. The AI response contains many factual errors or serious flaws, or is irrelevant to the user query, providing little to no value to the user.
- 3~4 points: Partially valuable/meaningful. The AI response contains several factual errors or serious flaws, or poorly meets the user's requirements, but has some redeeming qualities and offers partial value to the user.
- 5~6 points: Flawed. The AI response has some issues, such as minor factual errors/flaws, or does not fully meet the user's requirements. However, these are relatively minor, and the response generally satisfies the user's needs.
- 7~8 points: Meets requirements. The AI response satisfies the user's needs well, with no major flaws or errors, or only very minor issues that do not affect overall quality.
- 9~10 points: High quality. The AI response perfectly meets the user's requirements, with virtually no room for improvement.

## Output Format
First, analyze the query itself and understand the user's intent. Then provide your analysis of the model's response. Summarize your evaluation in two aspects: "Strengths" and "Weaknesses". Finally, write your score. The score should appear on the last line in the following format:
Score: [your score]
"""

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    evaluation_results = []
    scores = []
    
    full_checklist = general_checklist + text_reference
    
    for i, response in enumerate(transcriptions):
        
        prompt = meta_prompt.format(
            query=user_query,
            response=response,
            checklist=full_checklist
        )
        
        try:
            result = generate_text_chat(
                client=client,
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=0.5,
                top_p=0.95,
                n=1
            ).choices[0].message.content.strip()
            
            evaluation_results.append(result)
            score = extract_rating(result)
            scores.append(score)
            
        except Exception as e:
            print(f"    Error: {e}")
            evaluation_results.append(f"Error: {e}")
            scores.append(None)
    
    # Step 3: Calculate average and normalize
    valid_scores = [s for s in scores if s is not None]
    
    if not valid_scores:
        return 0.0, {
            'error': 'No valid GPT scores obtained',
            'error_type': 'gpt_evaluation_failed',
            'transcriptions': transcriptions,
            'evaluations': evaluation_results,
            'scores': scores
        }
    
    avg_gpt_score = sum(valid_scores) / len(valid_scores)
    
    # Normalize: (score - 1) / 9.0  [1-10 scale -> 0-1]
    normalized_score = min(max(0.0, (avg_gpt_score - 1) / 9.0), 1.0)
    
    # Prepare metadata
    metadata = {
        'transcriptions': transcriptions,
        'evaluations': evaluation_results,
        'gpt_raw_scores': scores,
        'gpt_avg_score': avg_gpt_score,
        'normalized_score': normalized_score,
        'user_query': user_query,
        'checklist': text_reference,
        'transcription_metadata': whisper_metadata
    }
    
    return normalized_score, metadata


def wildspeech_utmos(
    model_response_audio_path: str
) -> Tuple[float, Dict[str, Any]]:
    """
    WildSpeech-Bench UTMOS evaluation (exact reproduction)
    
    Uses UTMOS22 model to evaluate audio quality
    - Single evaluation (not 3 times like GPT)
    - Normalizes from 1-5 scale to [0, 1]
    
    Args:
        model_response_audio_path: Path to model's audio output
        
    Returns:
        Tuple of (score, metadata)
        - score: UTMOS score normalized to [0, 1] using (mos - 1) / 4
        - metadata: dict with evaluation details
    """
    try:
        import torch
        import librosa
        
        # Load UTMOS22 model (exactly as in WildSpeech-Bench)
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True
        )
        
        # Load audio
        wave, sr = librosa.load(model_response_audio_path, sr=None, mono=True)
        
        # Predict MOS score
        mos_score = float(predictor(torch.from_numpy(wave).unsqueeze(0), sr).item())
        
        # Normalize: (mos - 1.0) / 4.0  [1-5 scale -> 0-1]
        normalized_score = min(max(0.0, (mos_score - 1.0) / 4.0), 1.0)
        
        metadata = {
            'audio_path': model_response_audio_path,
            'mos_raw_score': mos_score,
            'normalized_score': normalized_score
        }
        
        return normalized_score, metadata
        
    except Exception as e:
        print(f"  - Error calculating MOS: {e}")
        # Use placeholder as in original code
        mos_score = 3.5
        normalized_score = (mos_score - 1.0) / 4.0
        
        return normalized_score, {
            'error': f'UTMOS evaluation failed: {str(e)}',
            'error_type': 'utmos_evaluation_error',
            'audio_path': model_response_audio_path,
            'mos_raw_score': mos_score,
            'normalized_score': normalized_score,
            'note': 'Using placeholder score 3.5'
        }