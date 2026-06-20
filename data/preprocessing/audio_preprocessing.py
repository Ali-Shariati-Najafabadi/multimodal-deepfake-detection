import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    def __init__(
        self,
        target_sr: int = 16000,
        segment_duration: float = 2.0,
        overlap: float = 0.5,
        normalize: bool = True
    ):
        self.target_sr = target_sr
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.normalize = normalize
        
        self.segment_length = int(segment_duration * target_sr)
        self.hop_length = int(self.segment_length * (1.0 - overlap))
    
    
    def extract_audio_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> Tuple[np.ndarray, int]:
        try:
            import av
            
            container = av.open(video_path)
            audio_stream = container.streams.audio[0]
            
            audio_data = []
            for frame in container.decode(audio=0):
                audio_data.append(frame.to_ndarray())
            
            if len(audio_data) == 0:
                raise ValueError("No audio found")
            
            audio = np.concatenate(audio_data, axis=1)[0]
            sr = audio_stream.rate
            
            container.close()
            
            if output_path:
                sf.write(output_path, audio, sr)
            
            return audio, sr
            
        except Exception as e:
            raise ValueError(f"Cannot extract audio: {e}")
    
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=self.target_sr)
    
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / (rms * 10)
        return np.clip(audio, -1.0, 1.0)
    
    
    def segment_audio(
        self,
        audio: np.ndarray,
        pad_last: bool = True
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        total_length = len(audio)
        segments = []
        timestamps = []
        start = 0
        while start < total_length:
            end = start + self.segment_length
            
            if end <= total_length:
                segment = audio[start:end]
            else:
                if pad_last:
                    segment = np.zeros(self.segment_length, dtype=audio.dtype)
                    remaining = total_length - start
                    segment[:remaining] = audio[start:]
                else:
                    break
            segments.append(torch.from_numpy(segment).float())
            start_time = start / self.target_sr
            end_time = min(end, total_length) / self.target_sr
            timestamps.append((start_time, end_time))
            start += self.hop_length
        
        if len(segments) == 0:
            segments.append(torch.zeros(self.segment_length))
            timestamps.append((0.0, self.segment_duration))
        return torch.stack(segments), timestamps
    
    
    def process_audio(
        self,
        video_path: str,
        start_time: float = None,
        duration: float = None
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]], float]:
        try:
            audio, orig_sr = self.extract_audio_from_video(video_path)
            
            # ✅ Crop audio only if both start_time and duration provided
            if start_time is not None and duration is not None:
                start_sample = int(start_time * orig_sr)
                duration_samples = int(duration * orig_sr)
                end_sample = start_sample + duration_samples
                
                if start_sample < len(audio):
                    audio = audio[start_sample:end_sample]
                    
                    # Pad if shorter than requested
                    if len(audio) < duration_samples:
                        audio = np.pad(audio, (0, duration_samples - len(audio)))
                else:
                    # Start beyond audio length
                    audio = np.zeros(duration_samples, dtype=audio.dtype)
            else:
                # ✅ No cropping: Use full audio (up to max)
                # Optionally limit to max_duration (e.g., 16s)
                max_samples = int(16.0 * orig_sr)  # 16 seconds max
                if len(audio) > max_samples:
                    audio = audio[:max_samples]
            
            # Resample to target sample rate
            audio = self.resample_audio(audio, orig_sr)
            
            # Normalize if enabled
            if self.normalize:
                audio = self.normalize_audio(audio)
            
            # Segment into overlapping chunks
            segments, timestamps = self.segment_audio(audio)
            duration = len(audio) / self.target_sr
            
            return segments, timestamps, duration
            
        except Exception as e:
            # Fallback: return single zero segment
            segments = torch.zeros(1, self.segment_length)
            timestamps = [(0.0, self.segment_duration)]
            return segments, timestamps, self.segment_duration




    
    
    def process_audio_batch(
        self,
        video_paths: List[str]
    ) -> List[Tuple[torch.Tensor, List[Tuple[float, float]], float]]:
        results = []
        for video_path in video_paths:
            try:
                result = self.process_audio(video_path)
                results.append(result)
            except Exception as e:
                empty_segment = torch.zeros(1, self.segment_length)
                results.append((empty_segment, [(0.0, self.segment_duration)], 0.0))
        return results