from typing import Dict, Any, Tuple, Optional
import os
from openai import OpenAI


class WhisperTranscriber:
    """
    Whisper transcriber using OpenAI API
    """

    def __init__(self, model_name: str = "whisper-1", api_key: Optional[str] = None):
        """
        Initialize Whisper transcriber with OpenAI API
        
        Args:
            model_name: OpenAI Whisper model name (default: whisper-1)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided)
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        #print(f"Initialized Whisper transcriber with OpenAI API: {model_name}")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text using OpenAI API.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        try:
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file
                )
            
            text = response.text.strip()
            return text
        except Exception as e:
            print(f"Transcription failed for {audio_path}: {e}")
            return ""


def whisper_transcribe(
    audio_path: str,
    transcriber: Optional[WhisperTranscriber] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Transcribe audio using OpenAI Whisper API
    
    Args:
        audio_path: Path to audio file
        transcriber: WhisperTranscriber instance (will create if None)
        
    Returns:
        Tuple of (transcribed_text, metadata)
    """
    # Create transcriber if not provided
    if transcriber is None:
        transcriber = WhisperTranscriber()
    
    try:
        transcribed_text = transcriber.transcribe(audio_path)
        
        metadata = {
            'audio_path': audio_path,
            'transcribed_text': transcribed_text,
        }
        
        return transcribed_text, metadata
        
    except Exception as e:
        print(f"Whisper transcription failed: {str(e)}")
        return "", {
            'error': f"Whisper transcription failed: {str(e)}",
            'error_type': 'transcription_error',
            'audio_path': audio_path
        }