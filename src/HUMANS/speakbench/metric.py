import os
import json
import base64
import wave
import audioop
import time
from io import BytesIO
from typing import Dict, Tuple, Optional, List, Any
from collections import defaultdict, Counter
from pathlib import Path

# Get the directory where this metric file is located (speakbench folder)
PACKAGE_DIR = Path(__file__).parent
DEFAULT_SIGNAL_FOLDER = str(PACKAGE_DIR / "signal_audios")


def encode_audio_file(audio_path: str) -> str:
    """Encode audio file to base64 string."""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def generate_signal_audio(text: str, output_path: str):
    """Generate TTS audio for signals like 'Instruction', 'Audio 1', etc."""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    if not os.path.exists(output_path):
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="coral",
            input=text,
            instructions="Speak in a clear and instructive tone",
            response_format="wav",
        ) as response:
            response.stream_to_file(output_path)
    
    return output_path


def concatenate_audio_files(
    instruction_path: str,
    audio1_path: str,
    audio2_path: str,
    output_path: str,
    signal_folder: str = None,
) -> str:
    """
    Concatenate audio files with spoken signals between them.
    Pattern: "Test" -> silence -> "Instruction" -> silence -> instruction_audio -> silence ->
             "Audio 1" -> silence -> audio1 -> silence -> "Audio 2" -> silence -> audio2 -> silence
    """
    # Use package default if not specified
    if signal_folder is None:
        signal_folder = DEFAULT_SIGNAL_FOLDER
    
    os.makedirs(signal_folder, exist_ok=True)
    
    audio_paths = [instruction_path, audio1_path, audio2_path]
    
    # Determine target sample rate
    sample_rates = []
    for audio_path in audio_paths:
        with wave.open(audio_path, "rb") as w:
            sample_rates.append(w.getframerate())
    
    target_sample_rate = Counter(sample_rates).most_common(1)[0][0]
    
    # Get audio parameters from first file
    with wave.open(audio_paths[0], "rb") as first_file:
        params = first_file.getparams()
        params = params._replace(framerate=target_sample_rate)
        nchannels = params.nchannels
        sampwidth = params.sampwidth
    
    # Generate signal audio files
    signal_files = {
        "test.wav": "Test",
        "instruction.wav": "Instruction",
        "audio_1.wav": "Audio 1",
        "audio_2.wav": "Audio 2",
    }
    
    signal_segments = {}
    for filename, text in signal_files.items():
        signal_path = os.path.join(signal_folder, filename)
        generate_signal_audio(text, signal_path)
        
        # Resample signal to match target sample rate
        with wave.open(signal_path, "rb") as w:
            signal_rate = w.getframerate()
            signal_frames = w.readframes(w.getnframes())
            signal_channels = w.getnchannels()
            signal_width = w.getsampwidth()
            
            if signal_rate != target_sample_rate:
                signal_frames, _ = audioop.ratecv(
                    signal_frames,
                    signal_width,
                    signal_channels,
                    signal_rate,
                    target_sample_rate,
                    None,
                )
            
            signal_segments[filename] = signal_frames
    
    # Function to add silence
    def add_silence(duration=0.5):
        return b"\x00" * (int(duration * target_sample_rate) * sampwidth * nchannels)
    
    # Create concatenated audio file
    with wave.open(output_path, "wb") as output_file:
        output_file.setparams(params)
        
        # Add "Test" signal
        output_file.writeframes(signal_segments["test.wav"])
        output_file.writeframes(add_silence())
        
        # Add "Instruction" signal
        output_file.writeframes(signal_segments["instruction.wav"])
        output_file.writeframes(add_silence())
        
        # Add instruction audio
        with wave.open(instruction_path, "rb") as w:
            audio_rate = w.getframerate()
            audio_frames = w.readframes(w.getnframes())
            audio_channels = w.getnchannels()
            audio_width = w.getsampwidth()
            
            if audio_rate != target_sample_rate:
                audio_frames, _ = audioop.ratecv(
                    audio_frames, audio_width, audio_channels,
                    audio_rate, target_sample_rate, None,
                )
            
            output_file.writeframes(audio_frames)
            output_file.writeframes(add_silence())
        
        # Add "Audio 1" signal
        output_file.writeframes(signal_segments["audio_1.wav"])
        output_file.writeframes(add_silence())
        
        # Add audio 1
        with wave.open(audio1_path, "rb") as w:
            audio_rate = w.getframerate()
            audio_frames = w.readframes(w.getnframes())
            audio_channels = w.getnchannels()
            audio_width = w.getsampwidth()
            
            if audio_rate != target_sample_rate:
                audio_frames, _ = audioop.ratecv(
                    audio_frames, audio_width, audio_channels,
                    audio_rate, target_sample_rate, None,
                )
            
            output_file.writeframes(audio_frames)
            output_file.writeframes(add_silence())
        
        # Add "Audio 2" signal
        output_file.writeframes(signal_segments["audio_2.wav"])
        output_file.writeframes(add_silence())
        
        # Add audio 2
        with wave.open(audio2_path, "rb") as w:
            audio_rate = w.getframerate()
            audio_frames = w.readframes(w.getnframes())
            audio_channels = w.getnchannels()
            audio_width = w.getsampwidth()
            
            if audio_rate != target_sample_rate:
                audio_frames, _ = audioop.ratecv(
                    audio_frames, audio_width, audio_channels,
                    audio_rate, target_sample_rate, None,
                )
            
            output_file.writeframes(audio_frames)
            output_file.writeframes(add_silence())
    
    return output_path


# System prompts for the 3 CoT types
SYSTEM_PROMPTS = {
    "lexical_cot": (
        "You are an evaluator of audio outputs produced by different audio-capable large language models. Your task is to compare two audio responses (Audio 1 and Audio 2) generated according to a user's instruction. "
        "Focus EXCLUSIVELY on the lexical content (the actual words and language used) and COMPLETELY IGNORE all of the following: "
        "- Pronunciation or enunciation of words "
        "- Speaking style, cadence, or rhythm "
        "- Emotional tone or expressiveness "
        "- Voice pitch, volume, or speed "
        "- Accents or speech patterns "
        "- Non-linguistic sounds or effects "
        "- Any other audio qualities "
        "Evaluate based on these criteria ONLY: "
        "1. Accuracy: Does the textual content correctly address what was requested? "
        "2. Completeness: Does the response include all the information needed to fulfill the request? "
        "3. Organization: Is the content structured in a clear, logical manner? "
        "4. Language use: Is the vocabulary and phrasing appropriate for the task? "
        "IMPORTANT: Even for tasks primarily focused on pronunciation, accents, or tones (like demonstrating Chinese tones), evaluate ONLY the textual content as if you were reading a transcript. Do NOT consider how well the model actually pronounced anything. "
        "Follow this process: "
        "1. Analyze what information was requested in the user's instruction "
        "2. Evaluate Audio 1's lexical content only (as if reading a transcript) "
        "3. Evaluate Audio 2's lexical content only (as if reading a transcript) "
        "4. Compare their strengths and weaknesses in terms of text content alone "
        "5. Decide which has better lexical content overall "
        "Pretend you are evaluating written transcripts rather than audio, and focus solely on what words were chosen. Avoid position bias and don't let response length influence your evaluation. After your analysis, output valid JSON with exactly two keys: "
        "'reasoning' (your explanation of the comparison) and 'label' (a string value: '1' if the first audio is better, '2' if the second audio is better, or 'tie' if they are equally good/bad. Please use \"tie\" sparingly, and only when you absolutely cannot choose the winner.)"
    ),
    "paralinguistic_cot": (
        "You are an evaluator of audio outputs produced by different audio-capable large language models. Your task is to compare two audio responses (Audio 1 and Audio 2) generated according to a user's instruction. "
        "Focus EXCLUSIVELY on paralinguistic features (how things are said) and ignore the lexical content (what words are used). "
        "Evaluate based on these criteria: "
        "1. Tone: Does the voice express the appropriate emotion, mood, or attitude? "
        "2. Prosody: How well does the response use rhythm, stress, intonation, and pacing? "
        "3. Expressiveness: Does the voice convey emphasis, contrast, and nuance appropriately? "
        "4. Accent/Pronunciation: If requested, how well does the response match the requested accent or pronunciation pattern? "
        "For tasks involving demonstration of tones, accents or specific speech patterns (like Chinese tones), focus entirely on how well these specific paralinguistic features were executed. "
        "Follow this process: "
        "1. Analyze what paralinguistic features were requested in the user's instruction "
        "2. Evaluate Audio 1's paralinguistic features only "
        "3. Evaluate Audio 2's paralinguistic features only "
        "4. Compare their strengths and weaknesses in paralinguistic execution "
        "5. Decide which has better paralinguistic features overall "
        "Avoid position bias and don't let content quality influence your evaluation. After your analysis, output valid JSON with exactly two keys: "
        "'reasoning' (your explanation of the comparison) and 'label' (a string value: '1' if the first audio is better, '2' if the second audio is better, or 'tie' if they are equally good/bad. Please use \"tie\" sparingly, and only when you absolutely cannot choose the winner.)"
    ),
    "speech_quality_cot": (
        "You are an evaluator of audio outputs produced by different audio-capable large language models. Your task is to compare two audio responses (Audio 1 and Audio 2) generated according to a user's instruction. "
        "Focus EXCLUSIVELY on technical speech quality aspects and ignore both content and expressive features. "
        "Evaluate based on these criteria: "
        "1. Clarity: How clear and intelligible is the speech? "
        "2. Naturalness: How natural does the voice sound (vs robotic or artificial)? "
        "3. Fluency: Is the speech smooth with appropriate pauses, or are there unnatural breaks, stutters, or glitches? "
        "4. Pronunciation: Are words pronounced correctly (regardless of accent)? "
        "5. Audio quality: Is the speech free from distortions, artifacts, or background noise? "
        "Follow this process: "
        "1. Analyze what speech quality features might be relevant to the user's instruction "
        "2. Evaluate Audio 1's speech quality features only "
        "3. Evaluate Audio 2's speech quality features only "
        "4. Compare their strengths and weaknesses in speech quality "
        "5. Decide which has better speech quality overall "
        "Avoid position bias and don't let content or expressiveness influence your evaluation. After your analysis, output valid JSON with exactly two keys: "
        "'reasoning' (your explanation of the comparison) and 'label' (a string value: '1' if the first audio is better, '2' if the second audio is better, or 'tie' if they are equally good/bad. Please use \"tie\" sparingly, and only when you absolutely cannot choose the winner.)"
    ),
}


def get_gpt_prompt(
    prompt_type: str,
    instruction_path: str,
    audio1_path: str,
    audio2_path: str,
    concat_test: bool = True,
) -> list:
    """Create prompt for GPT-4o-audio-preview."""
    
    system_prompt = SYSTEM_PROMPTS[prompt_type]
    
    user_message = (
        f"Please analyze which of the two recordings follows the instruction better, or tie. "
        f"Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if concat_test:
        os.makedirs("humans-audio", exist_ok=True)
        concat_path = os.path.join("humans-audio", f"concat_test_{prompt_type}_{time.time()}.wav")
        concatenate_audio_files(instruction_path, audio1_path, audio2_path, concat_path)
        
        test_encoded = encode_audio_file(concat_path)
        
        user_content = [
            {"type": "text", "text": "Please analyze these audio clips:"},
            {
                "type": "input_audio",
                "input_audio": {"data": test_encoded, "format": "wav"},
            },
            {"type": "text", "text": user_message}
        ]
        
        os.remove(concat_path)
    else:
        instruction_encoded = encode_audio_file(instruction_path)
        audio1_encoded = encode_audio_file(audio1_path)
        audio2_encoded = encode_audio_file(audio2_path)
        
        user_content = [
            {"type": "text", "text": "Here is the instruction for this test:"},
            {
                "type": "input_audio",
                "input_audio": {"data": instruction_encoded, "format": "wav"},
            },
            {"type": "text", "text": "Here is the first audio clip:"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio1_encoded, "format": "wav"},
            },
            {"type": "text", "text": "Here is the second audio clip:"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio2_encoded, "format": "wav"},
            },
            {"type": "text", "text": user_message}
        ]
    
    messages.append({"role": "user", "content": user_content})
    return messages


def get_gemini_prompt(
    prompt_type: str,
    instruction_path: str,
    audio1_path: str,
    audio2_path: str,
    concat_test: bool = True,
) -> list:
    """Create prompt for Gemini."""
    from pydub import AudioSegment
    from google.genai import types 
    
    system_prompt = SYSTEM_PROMPTS[prompt_type]
    
    user_message = (
        f"Please analyze which of the two recordings follows the instruction better, or tie. "
        f"Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
    )
    
    messages = [system_prompt]
    
    if concat_test:
        os.makedirs("humans-audio", exist_ok=True)
        concat_path = os.path.join("humans-audio", f"concat_test_{prompt_type}_{time.time()}.wav")
        concatenate_audio_files(instruction_path, audio1_path, audio2_path, concat_path)
        
        test_audio = AudioSegment.from_file(concat_path)
        if test_audio.frame_rate != 16000:
            test_audio = test_audio.set_frame_rate(16000)
        
        test_content = BytesIO()
        test_audio.export(test_content, format="wav")
        test_content.seek(0)
        
        messages.append("Please analyze these audio clips:")
        messages.append(types.Part.from_bytes(
            data=test_content.read(),
            mime_type='audio/wav'
        ))
        
        os.remove(concat_path)
    else:
        instruction_audio = AudioSegment.from_file(instruction_path)
        if instruction_audio.frame_rate != 16000:
            instruction_audio = instruction_audio.set_frame_rate(16000)
        instruction_content = BytesIO()
        instruction_audio.export(instruction_content, format="wav")
        instruction_content.seek(0)
        
        audio1 = AudioSegment.from_file(audio1_path)
        if audio1.frame_rate != 16000:
            audio1 = audio1.set_frame_rate(16000)
        audio1_content = BytesIO()
        audio1.export(audio1_content, format="wav")
        audio1_content.seek(0)
        
        audio2 = AudioSegment.from_file(audio2_path)
        if audio2.frame_rate != 16000:
            audio2 = audio2.set_frame_rate(16000)
        audio2_content = BytesIO()
        audio2.export(audio2_content, format="wav")
        audio2_content.seek(0)
        
        messages.append("Here is the instruction for this test:")
        messages.append(types.Part.from_bytes(
            data=instruction_content.read(),
            mime_type='audio/wav'
        ))
        messages.append("Here is the first audio clip:")
        messages.append(types.Part.from_bytes(
            data=audio1_content.read(),
            mime_type='audio/wav'
        ))
        messages.append("Here is the second audio clip:")
        messages.append(types.Part.from_bytes(
            data=audio2_content.read(),
            mime_type='audio/wav'
        ))
    
    messages.append(user_message)
    return messages


def get_model_response_gpt(messages: list) -> Optional[Tuple[str, str]]:
    """Get response from GPT-4o-audio-preview."""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=messages,
        )
        
        response_text = response.choices[0].message.content
        return (response.id, response_text)
    except Exception as e:
        print(f"Error getting GPT response: {e}")
        return None


def get_model_response_gemini(messages: list) -> Optional[Tuple[str, str]]:
    """Get response from Gemini."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai not installed. Run: pip install google-genai")
        return None
    
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages
        )
        return ("gemini_response", response.text)
    except Exception as e:
        print(f"Error getting Gemini response: {e}")
        return None


def extract_json_from_response(response_text: str) -> Optional[dict]:
    """Extract JSON from response text."""
    import re
    
    json_match = None
    
    # Try to find JSON between code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = list(re.finditer(code_block_pattern, response_text, re.DOTALL))
    if matches:
        json_match = matches[0].group(1)
    
    # Try to find JSON object directly
    if not json_match:
        json_pattern = r'\{[^{}]*"label"[^{}]*\}'
        matches = list(re.finditer(json_pattern, response_text, re.DOTALL))
        if matches:
            json_match = matches[0].group(0)
    
    # Try parsing the entire response as JSON
    if not json_match:
        json_match = response_text.strip()
    
    try:
        return json.loads(json_match)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from: {json_match}")
        return None


def convert_prediction_to_numeric(pred, position=1):
    """
    Convert string prediction to numeric value.
    
    Args:
        pred: Prediction label ("1", "2", or "tie")
        position: 1 if model is audio1, 2 if model is audio2
        
    Returns:
        Numeric score where 0.0 = model wins, 0.5 = tie, 1.0 = model loses
    """
    if pred == "tie":
        return 0.5
    
    if position == 1:
        # Position 1: model is audio1, gpt4o is audio2
        if pred == "1":
            return 0.0  # model wins
        elif pred == "2":
            return 1.0  # model loses
    else:
        # Position 2: gpt4o is audio1, model is audio2
        if pred == "1":
            return 1.0  # model loses (gpt4o won)
        elif pred == "2":
            return 0.0  # model wins
    
    return None


def convert_numeric_to_prediction(value):
    """Convert numeric value to prediction string."""
    if value < 0.5:
        return "1"
    elif value > 0.5:
        return "2"
    else:
        return "tie"


def convert_prediction_to_result(prediction, position):
    """Convert a prediction to a result (win/loss/tie)."""
    if position == 1:
        return {"1": "win", "2": "loss", "tie": "tie"}.get(prediction, "unknown")
    else:
        return {"1": "loss", "2": "win", "tie": "tie"}.get(prediction, "unknown")


def speakbench_WinRate(
    model_response_audio_path: str,
    audio_input_path: str,
    audio_reference_path: str,
    concat_test: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    """
    SpeakBench WinRate evaluation metric
    
    Compares model's audio output against GPT-4o-audio-preview's reference using
    ensemble of 3 CoT prompts (lexical, paralinguistic, speech_quality) with
    2 positions each (6 API calls total).
    
    Args:
        audio_input_path: Path to instruction audio (user's request)
        audio_reference_path: Path to GPT-4o-audio-preview's reference audio
        model_response_audio_path: Path to model's audio response
        concat_test: Whether to concatenate test audio files
        
    Returns:
        Tuple of (score, metadata)
        - score: 1.0 if model wins, 0.5 if tie, 0.0 if loss
        - metadata: dict with detailed evaluation results
    """
    
    # Determine which judge model to use
    use_gemini = "GOOGLE_API_KEY" in os.environ
    judge_model = "gemini-2.5-flash" if use_gemini else "gpt-4o-audio-preview"
    
    prompt_types = ["lexical_cot", "paralinguistic_cot", "speech_quality_cot"]
    
    # Store predictions for ensemble
    position1_predictions = []
    position2_predictions = []
    
    # Store detailed results for each prompt type
    detailed_results = {}
    
    for prompt_type in prompt_types:
        # Position 1: model_audio as audio1, gpt4o_audio as audio2
        if use_gemini:
            messages_pos1 = get_gemini_prompt(
                prompt_type, audio_input_path, model_response_audio_path, 
                audio_reference_path, concat_test
            )
            response_data_pos1 = get_model_response_gemini(messages_pos1)
        else:
            messages_pos1 = get_gpt_prompt(
                prompt_type, audio_input_path, model_response_audio_path, 
                audio_reference_path, concat_test
            )
            response_data_pos1 = get_model_response_gpt(messages_pos1)
        
        # Position 2: gpt4o_audio as audio1, model_audio as audio2
        if use_gemini:
            messages_pos2 = get_gemini_prompt(
                prompt_type, audio_input_path, audio_reference_path, 
                model_response_audio_path, concat_test
            )
            response_data_pos2 = get_model_response_gemini(messages_pos2)
        else:
            messages_pos2 = get_gpt_prompt(
                prompt_type, audio_input_path, audio_reference_path, 
                model_response_audio_path, concat_test
            )
            response_data_pos2 = get_model_response_gpt(messages_pos2)
        
        # Check if responses are valid
        if response_data_pos1 is None or response_data_pos2 is None:
            print(f"    ERROR: Failed to get response for {prompt_type}")
            continue
        
        _, prediction_text_pos1 = response_data_pos1
        _, prediction_text_pos2 = response_data_pos2
        
        prediction_json_pos1 = extract_json_from_response(prediction_text_pos1)
        prediction_json_pos2 = extract_json_from_response(prediction_text_pos2)
        
        if prediction_json_pos1 is None or prediction_json_pos2 is None:
            print(f"    ERROR: Failed to extract JSON for {prompt_type}")
            continue
        
        prediction_pos1 = prediction_json_pos1.get("label", None)
        prediction_pos2 = prediction_json_pos2.get("label", None)
        
        reasoning_pos1 = prediction_json_pos1.get("reasoning", "")
        reasoning_pos2 = prediction_json_pos2.get("reasoning", "")
        
        if prediction_pos1 is None or prediction_pos2 is None:
            print(f"    ERROR: No prediction field for {prompt_type}")
            continue
        
        # Convert to numeric for ensemble
        numeric_pos1 = convert_prediction_to_numeric(prediction_pos1, position=1)
        numeric_pos2 = convert_prediction_to_numeric(prediction_pos2, position=2)
        
        if numeric_pos1 is not None and numeric_pos2 is not None:
            position1_predictions.append(numeric_pos1)
            position2_predictions.append(numeric_pos2)
        
        # Interpret results
        result_pos1 = convert_prediction_to_result(prediction_pos1, 1)
        result_pos2 = convert_prediction_to_result(prediction_pos2, 2)
        
        # Store detailed results
        detailed_results[prompt_type] = {
            "position1_prediction": prediction_pos1,
            "position1_result": result_pos1,
            "position1_reasoning": reasoning_pos1,
            "position2_prediction": prediction_pos2,
            "position2_result": result_pos2,
            "position2_reasoning": reasoning_pos2,
        }
    
    # Calculate ensemble predictions
    if not position1_predictions or not position2_predictions:
        return 0.0, {
            "error": "Failed to get predictions from all prompt types",
            "error_type": "ensemble_evaluation_failed",
            "judge_model": judge_model,
            "detailed_results": detailed_results,
            "audio_input_path": audio_input_path,
            "audio_reference_path": audio_reference_path,
            "model_response_audio_path": model_response_audio_path,
        }
    
    # Average predictions across all prompt types
    avg_pos1 = sum(position1_predictions) / len(position1_predictions)
    avg_pos2 = sum(position2_predictions) / len(position2_predictions)
    
    # Convert back to prediction strings
    ensemble_pos1 = convert_numeric_to_prediction(avg_pos1)
    ensemble_pos2 = convert_numeric_to_prediction(avg_pos2)
    
    # Convert to results
    ensemble_result_pos1 = convert_prediction_to_result(ensemble_pos1, 1)
    ensemble_result_pos2 = convert_prediction_to_result(ensemble_pos2, 2)
    
    # Calculate final score: 1.0 if win, 0.5 if tie, 0.0 if loss
    # We use position1 where model is audio1
    if ensemble_result_pos1 == "win":
        final_score = 1.0
    elif ensemble_result_pos1 == "tie":
        final_score = 0.5
    else:  # loss
        final_score = 0.0
    
    # Prepare metadata
    metadata = {
        "judge_model": judge_model,
        "ensemble": {
            "position1_prediction": ensemble_pos1,
            "position1_result": ensemble_result_pos1,
            "position1_avg_numeric": avg_pos1,
            "position2_prediction": ensemble_pos2,
            "position2_result": ensemble_result_pos2,
            "position2_avg_numeric": avg_pos2,
            "num_voters": len(position1_predictions),
        },
        "detailed_results": detailed_results,
        "audio_input_path": audio_input_path,
        "audio_reference_path": audio_reference_path,
        "model_response_audio_path": model_response_audio_path,
    }
    
    return final_score, metadata