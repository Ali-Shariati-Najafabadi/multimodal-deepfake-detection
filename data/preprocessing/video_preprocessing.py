import cv2
import torch
import numpy as np
from pathlib import Path
#from facenet_pytorch import MTCNN
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
        
        # DIESE ZEILEN AUSKOMMENTIEREN (Zeile 28-36):
        # self.mtcnn = MTCNN(
        #     image_size=face_size,
        #     margin=0,
        #     min_face_size=min_face_size,
        #     thresholds=[0.6, 0.7, 0.7],
        #     factor=0.709,
        #     post_process=True,
        #     device=device,
        #     keep_all=False
        # )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])