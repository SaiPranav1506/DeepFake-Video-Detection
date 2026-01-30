"""
Multi-Agent System for Deepfake Detection
Integrates autonomous agents with the pretrained model for intelligent decision-making
"""

import torch
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from abc import ABC, abstractmethod
from src.pretrained_detector import PretrainedBackboneDetector
from torch.utils.data import DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    SAFE = 0
    WARNING = 1
    DANGER = 2
    CRITICAL = 3


@dataclass
class PredictionResult:
    """Structured prediction result"""
    video_id: str
    is_fake: bool
    confidence: float
    alert_level: AlertLevel
    frame_scores: np.ndarray
    timestamp: datetime
    explanation: str


class Agent(ABC):
    """Base agent class"""
    
    def __init__(self, name: str):
        self.name = name
        self.history = []
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data through agent"""
        pass
    
    def log_action(self, action: str, result: Any):
        """Log agent action"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result
        })


class InferenceAgent(Agent):
    """
    Inference Agent: Runs predictions on video data
    Responsibility: Execute model inference with error handling
    """
    
    def __init__(
        self,
        model_path: str,
        backbone_name: str = "efficientnet_b0",
        device: str = "cpu"
    ):
        super().__init__("InferenceAgent")
        self.device = device
        
        # Load model
        self.model = PretrainedBackboneDetector(
            backbone_name=backbone_name,
            pretrained=False,
            num_classes=2,
            use_temporal_attention=True
        ).to(device).eval()
        
        try:
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            logger.info(f"[OK] Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference on frames
        
        Args:
            frames: (B, T, C, H, W) tensor
        
        Returns:
            logits: (B, 2) class logits
            frame_scores: (B, T) per-frame importance
        """
        with torch.no_grad():
            logits, frame_scores = self.model(frames)
        
        self.log_action("inference", {
            "batch_size": frames.shape[0],
            "num_frames": frames.shape[1],
            "output_shape": logits.shape
        })
        
        return logits, frame_scores


class DecisionAgent(Agent):
    """
    Decision Agent: Makes alert decisions based on model predictions
    Responsibility: Convert predictions to actionable decisions
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        high_confidence_threshold: float = 0.95
    ):
        super().__init__("DecisionAgent")
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
    
    def process(self, prediction: Dict) -> PredictionResult:
        """
        Make decision based on prediction
        
        Args:
            prediction: {
                'video_id': str,
                'logits': tensor (2,),
                'frame_scores': tensor (T,),
                'probs': tensor (2,)
            }
        
        Returns:
            PredictionResult with alert level
        """
        video_id = prediction['video_id']
        probs = prediction.get('probs')
        logits = prediction.get('logits')
        frame_scores = prediction.get('frame_scores')

        # Prefer the app's thresholded verdict when provided.
        # This prevents agent alerts from contradicting DETECT_FAKE_THRESHOLD.
        pred_class = prediction.get('pred_class', None)
        if pred_class in (0, 1):
            is_fake = bool(int(pred_class) == 1)
            try:
                confidence = float(prediction.get('confidence', 0.0))
            except Exception:
                confidence = 0.0
        else:
            # Fallback: infer verdict from probabilities (equivalent to a 0.5 threshold).
            if probs is None:
                raise ValueError("Missing 'probs' for DecisionAgent")
                try:
                    fake_idx = int(str(os.environ.get('FAKE_CLASS_INDEX', '1')).strip())
                except Exception:
                    fake_idx = 1
                fake_idx = 1 if fake_idx not in (0, 1) else fake_idx
                real_idx = 1 - int(fake_idx)
                is_fake = bool((probs[int(fake_idx)] > probs[int(real_idx)]).item())
            confidence = float(probs.max().item())

        if frame_scores is None:
            frame_scores = torch.zeros(8)
        
        # Determine alert level
        alert_level = self._determine_alert_level(is_fake, confidence, frame_scores)
        
        # Generate explanation
        explanation = self._generate_explanation(is_fake, confidence, frame_scores)
        
        result = PredictionResult(
            video_id=video_id,
            is_fake=is_fake,
            confidence=confidence,
            alert_level=alert_level,
            frame_scores=frame_scores.cpu().numpy(),
            timestamp=datetime.now(),
            explanation=explanation
        )
        
        self.log_action("decision", {
            "is_fake": is_fake,
            "confidence": confidence,
            "alert_level": alert_level.name
        })
        
        return result
    
    def _determine_alert_level(self, is_fake: bool, confidence: float, frame_scores: torch.Tensor) -> AlertLevel:
        """Determine alert severity"""
        if not is_fake:
            return AlertLevel.SAFE
        
        if confidence > self.high_confidence_threshold:
            return AlertLevel.CRITICAL
        elif confidence > self.confidence_threshold:
            return AlertLevel.DANGER
        else:
            return AlertLevel.WARNING
    
    def _generate_explanation(self, is_fake: bool, confidence: float, frame_scores: torch.Tensor) -> str:
        """Generate human-readable explanation"""
        if not is_fake:
            return f"Video appears authentic (confidence: {confidence:.1%})"
        
        # Find frames with highest fake scores
        top_frames = torch.topk(frame_scores, k=min(3, len(frame_scores)))
        
        if confidence > self.high_confidence_threshold:
            return f"CRITICAL: High-confidence deepfake detected ({confidence:.1%}). Suspicious activity in frames {top_frames.indices.tolist()}"
        elif confidence > self.confidence_threshold:
            return f"WARNING: Deepfake likely ({confidence:.1%}). Detected in frames {top_frames.indices.tolist()}"
        else:
            return f"UNCERTAIN: Possible deepfake ({confidence:.1%}). Low confidence - manual review recommended."


class MonitoringAgent(Agent):
    """
    Monitoring Agent: Tracks metrics and performance
    Responsibility: Log predictions, track accuracy, monitor anomalies
    """
    
    def __init__(self, output_dir: str = "logs/agent_monitoring"):
        super().__init__("MonitoringAgent")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = []
        self.metrics = {
            "total_processed": 0,
            "total_fake_detected": 0,
            "total_authentic": 0,
            "alerts_by_level": {level.name: 0 for level in AlertLevel}
        }
    
    def process(self, result: PredictionResult) -> Dict:
        """
        Log and monitor prediction result
        
        Args:
            result: PredictionResult from DecisionAgent
        
        Returns:
            Monitoring metrics
        """
        self.predictions.append(result)
        self.metrics["total_processed"] += 1
        
        if result.is_fake:
            self.metrics["total_fake_detected"] += 1
        else:
            self.metrics["total_authentic"] += 1
        
        self.metrics["alerts_by_level"][result.alert_level.name] += 1
        
        # Save to log
        self._save_prediction(result)
        
        self.log_action("monitoring", self.metrics.copy())
        
        return self.metrics
    
    def _save_prediction(self, result: PredictionResult):
        """Save prediction to log file"""
        log_file = self.output_dir / "predictions.jsonl"
        
        log_entry = {
            "timestamp": result.timestamp.isoformat(),
            "video_id": result.video_id,
            "is_fake": result.is_fake,
            "confidence": result.confidence,
            "alert_level": result.alert_level.name,
            "explanation": result.explanation
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_report(self) -> Dict:
        """Generate monitoring report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": self.metrics["total_processed"],
            "fake_percentage": (
                self.metrics["total_fake_detected"] / max(1, self.metrics["total_processed"])
                * 100
            ),
            "alerts": self.metrics["alerts_by_level"],
            "recent_predictions": [
                {
                    "video_id": p.video_id,
                    "is_fake": p.is_fake,
                    "confidence": p.confidence
                }
                for p in self.predictions[-10:]
            ]
        }


class ActionAgent(Agent):
    """
    Action Agent: Takes actions based on alerts
    Responsibility: Generate notifications, trigger workflows, file reports
    """
    
    def __init__(self, output_dir: str = "logs/agent_actions"):
        super().__init__("ActionAgent")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.actions_taken = []
    
    def process(self, result: PredictionResult) -> Dict:
        """
        Take action based on prediction result
        
        Args:
            result: PredictionResult from DecisionAgent
        
        Returns:
            Action summary
        """
        actions = []
        
        # Log based on alert level
        if result.alert_level == AlertLevel.SAFE:
            actions.append(self._log_safe(result))
        elif result.alert_level == AlertLevel.WARNING:
            actions.append(self._log_warning(result))
        elif result.alert_level == AlertLevel.DANGER:
            actions.append(self._alert_danger(result))
            actions.append(self._file_report(result))
        elif result.alert_level == AlertLevel.CRITICAL:
            actions.append(self._alert_critical(result))
            actions.append(self._file_report(result))
            actions.append(self._notify_admin(result))
        
        action_summary = {
            "video_id": result.video_id,
            "alert_level": result.alert_level.name,
            "actions_taken": actions,
            "timestamp": datetime.now().isoformat()
        }
        
        self.actions_taken.append(action_summary)
        self.log_action("action", action_summary)
        
        return action_summary
    
    def _log_safe(self, result: PredictionResult) -> str:
        """Log safe video"""
        msg = f"[SAFE] {result.video_id} - {result.explanation}"
        logger.info(msg)
        return msg
    
    def _log_warning(self, result: PredictionResult) -> str:
        """Log warning"""
        msg = f"[WARNING] {result.video_id} - {result.explanation}"
        logger.warning(msg)
        return msg
    
    def _alert_danger(self, result: PredictionResult) -> str:
        """Alert on danger level detection"""
        msg = f"[DANGER] {result.video_id} - {result.explanation}"
        logger.error(msg)
        return msg
    
    def _alert_critical(self, result: PredictionResult) -> str:
        """Alert on critical deepfake detection"""
        msg = f"[CRITICAL] {result.video_id} - {result.explanation}"
        logger.critical(msg)
        return msg
    
    def _file_report(self, result: PredictionResult) -> str:
        """File detailed report"""
        report_dir = self.output_dir / "reports"
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"{result.video_id}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            "video_id": result.video_id,
            "timestamp": result.timestamp.isoformat(),
            "is_fake": result.is_fake,
            "confidence": float(result.confidence),
            "alert_level": result.alert_level.name,
            "explanation": result.explanation,
            "top_suspicious_frames": result.frame_scores.argsort()[-3:].tolist()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return f"Report filed: {report_file}"
    
    def _notify_admin(self, result: PredictionResult) -> str:
        """Notify admin (placeholder for email/webhook)"""
        notification = {
            "alert_type": "CRITICAL_DEEPFAKE",
            "video_id": result.video_id,
            "confidence": float(result.confidence),
            "timestamp": result.timestamp.isoformat(),
            "message": result.explanation
        }
        
        # In production, send to:
        # - Email notification
        # - Slack webhook
        # - Database alert
        # - SMS alert
        
        logger.critical(f"ADMIN NOTIFICATION: {notification}")
        return "Admin notified"


class MultiAgentOrchestrator:
    """
    Orchestrator: Coordinates all agents
    Manages workflow and data flow between agents
    """
    
    def __init__(
        self,
        model_path: str,
        backbone_name: str = "efficientnet_b0",
        device: str = "cpu"
    ):
        self.device = device
        
        # Initialize agents
        self.inference_agent = InferenceAgent(model_path, backbone_name, device)
        self.decision_agent = DecisionAgent()
        self.monitoring_agent = MonitoringAgent()
        self.action_agent = ActionAgent()
        
        self.agents = [
            self.inference_agent,
            self.decision_agent,
            self.monitoring_agent,
            self.action_agent
        ]
        
        logger.info("[OK] Multi-Agent System Initialized")
        logger.info(f"  - Inference Agent: {self.inference_agent.name}")
        logger.info(f"  - Decision Agent: {self.decision_agent.name}")
        logger.info(f"  - Monitoring Agent: {self.monitoring_agent.name}")
        logger.info(f"  - Action Agent: {self.action_agent.name}")
    
    def process_video(
        self,
        frames: torch.Tensor,
        video_id: str
    ) -> Dict:
        """
        Process video through agent pipeline
        
        Args:
            frames: (B, T, C, H, W) video frames
            video_id: identifier for the video
        
        Returns:
            Final result with all agent outputs
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {video_id}")
        logger.info(f"{'='*70}")
        
        # Step 1: Inference
        logits, frame_scores = self.inference_agent.process(frames)
        
        probs = torch.softmax(logits[0], dim=0)
        
        # Step 2: Decision Making
        prediction_data = {
            'video_id': video_id,
            'logits': logits[0],
            'frame_scores': frame_scores[0],
            'probs': probs
        }
        decision_result = self.decision_agent.process(prediction_data)
        
        # Step 3: Monitoring
        metrics = self.monitoring_agent.process(decision_result)
        
        # Step 4: Action
        action_result = self.action_agent.process(decision_result)
        
        # Compile final result
        final_result = {
            "video_id": video_id,
            "inference": {
                "is_fake": decision_result.is_fake,
                "confidence": float(decision_result.confidence),
                "alert_level": decision_result.alert_level.name
            },
            "decision": {
                "explanation": decision_result.explanation,
                "frame_analysis": {
                    "top_frames": frame_scores[0].topk(3).indices.tolist(),
                    "scores": frame_scores[0].topk(3).values.tolist()
                }
            },
            "action": action_result,
            "monitoring": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[OK] Processing complete for {video_id}")
        logger.info(f"  Alert Level: {decision_result.alert_level.name}")
        logger.info(f"  Confidence: {decision_result.confidence:.1%}")
        
        return final_result
    
    def process_batch(
        self,
        frames_list: List[torch.Tensor],
        video_ids: List[str]
    ) -> List[Dict]:
        """Process multiple videos"""
        results = []
        for frames, video_id in zip(frames_list, video_ids):
            result = self.process_video(frames.unsqueeze(0), video_id)
            results.append(result)
        return results
    
    def get_system_report(self) -> Dict:
        """Generate system-wide report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "agents": [
                {
                    "name": agent.name,
                    "actions": len(agent.history),
                    "recent_actions": agent.history[-5:] if agent.history else []
                }
                for agent in self.agents
            ],
            "monitoring": self.monitoring_agent.get_report(),
            "recent_actions": self.action_agent.actions_taken[-10:]
        }
