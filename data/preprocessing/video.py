import cv2
import torch
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class VideoPreprocessor:
    def __init__(
        self,
        face_size: int = 224,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        min_face_size: int = 20,
        confidence_threshold: float = 0.7
    ):
        self.face_size = face_size
        self.device = device
        self.min_face_size = min_face_size
        self.confidence_threshold = confidence_threshold
        
        self.mtcnn = MTCNN(
            image_size=face_size,
            margin=0,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False
        )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    
    def extract_frames(self, video_path: str, sample_rate: int = 1) -> List[np.ndarray]:
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_rate == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames
    
    
    def detect_and_crop_face(
        self, 
        frame: np.ndarray,
        return_bbox: bool = False
    ) -> Optional[Tuple[torch.Tensor, Optional[list]]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            boxes, probs = self.mtcnn.detect(frame_rgb)
            
            if boxes is None or len(boxes) == 0:
                return (None, None) if return_bbox else None
            
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            prob = probs[best_idx]
            
            if prob < self.confidence_threshold:
                return (None, None) if return_bbox else None
            
            # Crop face
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_rgb.shape[1], x2)
            y2 = min(frame_rgb.shape[0], y2)
            
            face = frame_rgb[y1:y2, x1:x2]
            face = cv2.resize(face, (self.face_size, self.face_size))
            face_pil = Image.fromarray(face)
            
            face_tensor = self.transform(face_pil)
            
            if return_bbox:
                return face_tensor, [x1, y1, x2, y2]
            else:
                return face_tensor
            
        except Exception as e:
            return (None, None) if return_bbox else None
    
    
    def process_video(
        self,
        video_path: str,
        sample_rate: int = 1,
        max_frames: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        frames = self.extract_frames(video_path, sample_rate)
        
        if max_frames:
            frames = frames[:max_frames]
        
        face_crops = []
        valid_indices = []
        
        for idx, frame in enumerate(frames):
            face = self.detect_and_crop_face(frame)
            if face is not None:
                face_crops.append(face)
                valid_indices.append(idx * sample_rate)
        
        if len(face_crops) == 0:
            return torch.empty(0, 3, self.face_size, self.face_size), []
        
        return torch.stack(face_crops), valid_indices
    
    
    def process_video_batch(
        self,
        video_paths: List[str],
        sample_rate: int = 1,
        max_frames: Optional[int] = None
    ) -> List[Tuple[torch.Tensor, List[int]]]:
        results = []
        for video_path in video_paths:
            try:
                result = self.process_video(video_path, sample_rate, max_frames)
                results.append(result)
            except Exception as e:
                results.append((torch.empty(0, 3, self.face_size, self.face_size), []))
        return results
