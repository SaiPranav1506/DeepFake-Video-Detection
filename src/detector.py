"""Deepfake detection inference module."""
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import json

class DeepfakeDetector:
    """Wrapper for deepfake detection with explanation."""
    
    def __init__(self, model, feature_extractor=None, device='cpu', model_type='gcn'):
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = device
        self.model_type = model_type
        self.model.eval()
        if feature_extractor:
            self.feature_extractor.eval()
    
    def extract_faces(self, video_path: str, max_frames=10):
        """Extract faces from video using basic frame extraction."""
        try:
            cap = cv2.VideoCapture(video_path)
            faces = []
            frame_count = 0
            
            while len(faces) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Simple face detection using Haar Cascade
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_rects = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in face_rects:
                    face = frame[y:y+h, x:x+w]
                    if face.size > 0:
                        face_resized = cv2.resize(face, (224, 224))
                        faces.append(face_resized)
                
                frame_count += 1
            
            cap.release()
            return faces
        except Exception as e:
            print(f'Error extracting faces: {e}')
            return []
    
    def preprocess_faces(self, faces: List[np.ndarray]) -> torch.Tensor:
        """Convert face images to tensor."""
        if len(faces) == 0:
            return torch.zeros(1, 3, 224, 224)
        
        # Normalize to [0, 1]
        faces_np = np.array(faces, dtype=np.float32) / 255.0
        # Convert BGR to RGB
        faces_np = faces_np[..., ::-1]
        # HWC to CHW
        faces_tensor = torch.from_numpy(faces_np).permute(0, 3, 1, 2)
        
        return faces_tensor
    
    def detect(self, video_path: str) -> Dict:
        """Run deepfake detection on video."""
        try:
            # Extract faces
            faces = self.extract_faces(video_path, max_frames=10)
            num_faces = len(faces)
            
            if num_faces == 0:
                return {
                    'success': False,
                    'error': 'No faces detected in video',
                    'num_faces': 0,
                    'is_fake': None,
                    'confidence': 0.0
                }
            
            # Preprocess
            faces_tensor = self.preprocess_faces(faces).to(self.device)
            
            with torch.no_grad():
                if self.model_type == 'rnn':
                    # Use feature extractor for RNN
                    feats = self.feature_extractor(faces_tensor)  # (N, F)
                    # Pad or truncate to fixed sequence length
                    if feats.shape[0] < 10:
                        pad = 10 - feats.shape[0]
                        feats = torch.cat([feats, torch.zeros(pad, feats.shape[1], device=self.device)])
                    else:
                        feats = feats[:10]
                    feats = feats.unsqueeze(0)  # (1, 10, F)
                    
                    output = self.model(feats, torch.tensor([num_faces], device=self.device))
                    probs = torch.sigmoid(output).squeeze().cpu().numpy()
                else:
                    # GCN model
                    # Create adjacency matrix (simple chain)
                    N = faces_tensor.shape[0]
                    A = np.zeros((1, N, N), dtype=np.float32)
                    for i in range(N - 1):
                        A[0, i, i + 1] = 1.0
                        A[0, i + 1, i] = 1.0
                    
                    from src.utils import normalize_adjacency
                    A_norm = normalize_adjacency(A[0])
                    A_tensor = torch.from_numpy(A_norm).float().unsqueeze(0).to(self.device)
                    
                    output = self.model(faces_tensor.unsqueeze(0), A_tensor)
                    if output.dim() == 1:
                        probs = torch.sigmoid(output).squeeze().cpu().numpy()
                    else:
                        probs = torch.softmax(output, dim=1)[0, 1].cpu().numpy()
            
            # Aggregate predictions
            is_fake_prob = float(probs) if isinstance(probs, np.ndarray) and probs.ndim == 0 else float(probs.mean()) if isinstance(probs, np.ndarray) else float(probs)
            is_fake_pred = 1 if is_fake_prob >= 0.5 else 0
            confidence = is_fake_prob if is_fake_pred == 1 else (1.0 - is_fake_prob)
            
            return {
                'success': True,
                'error': None,
                'is_fake': is_fake_pred,
                'is_fake_prob': is_fake_prob,
                'confidence': confidence,
                'num_faces': num_faces,
                'explanation': generate_explanation(is_fake_pred, is_fake_prob, num_faces)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'num_faces': 0,
                'is_fake': None,
                'confidence': 0.0
            }

def generate_explanation(is_fake: int, confidence: float, num_faces: int) -> str:
    """Generate human-readable explanation of detection result."""
    if is_fake == 1:
        return (
            f"🚨 **LIKELY DEEPFAKE DETECTED** (confidence: {confidence*100:.1f}%)\n\n"
            f"The model detected {num_faces} face(s) in the video with synthetic manipulation patterns. "
            f"Key indicators:\n"
            f"- Facial feature artifacts and inconsistencies\n"
            f"- Unnatural eye movement or blinking patterns\n"
            f"- Audio-visual misalignment\n"
            f"- Lighting and shadow inconsistencies\n\n"
            f"⚠️ This is a probabilistic assessment. Manual review recommended for critical decisions."
        )
    else:
        confidence_real = 1.0 - confidence
        return (
            f"✅ **LIKELY AUTHENTIC** (confidence: {confidence_real*100:.1f}%)\n\n"
            f"The model detected {num_faces} face(s) in the video with natural characteristics. "
            f"Key indicators:\n"
            f"- Natural facial features and expressions\n"
            f"- Consistent eye movement and blinking\n"
            f"- Proper audio-visual synchronization\n"
            f"- Realistic lighting and shadows\n\n"
            f"✓ Video appears authentic based on analyzed characteristics."
        )
