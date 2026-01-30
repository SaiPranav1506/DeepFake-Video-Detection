"""
Enhanced Decision-Making Agent with Confidence Calibration
Integrates ensemble predictions with intelligent decision-making
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels based on confidence"""
    SAFE = 0           # Authentic (< 30% fake confidence)
    WARNING = 1        # Possible deepfake (30-70%)
    DANGER = 2         # Likely deepfake (70-95%)
    CRITICAL = 3       # High-confidence deepfake (> 95%)


@dataclass
class EnsemblePrediction:
    """Structured ensemble prediction result"""
    video_id: str
    is_fake: Optional[bool]
    confidence: float
    alert_level: AlertLevel
    ensemble_prob: float
    individual_probs: list  # Probabilities from each model
    frame_scores: np.ndarray
    uncertainty: float
    explanation: str


class EnhancedDecisionAgent:
    """
    Makes intelligent decisions from ensemble predictions
    with confidence calibration and uncertainty awareness
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        confidence_thresholds: Optional[Dict] = None,
        uncertainty_penalty: float = 0.1,
        fake_class_index: int = 1,
        # abstain policy params
        abstain_on_high_uncertainty: bool = True,
        abstain_uncertainty_threshold: float = 0.6,
        min_agreement_to_act: float = 0.6,
        decision_threshold: float = 0.5
    ):
        """
        Args:
            temperature: Temperature for softmax calibration
            confidence_thresholds: Custom thresholds for alert levels
            uncertainty_penalty: How much uncertainty reduces confidence
        """
        self.temperature = temperature
        self.uncertainty_penalty = uncertainty_penalty
        self.abstain_on_high_uncertainty = abstain_on_high_uncertainty
        self.abstain_uncertainty_threshold = abstain_uncertainty_threshold
        self.min_agreement_to_act = min_agreement_to_act
        self.decision_threshold = decision_threshold
        try:
            self.fake_class_index = int(fake_class_index)
        except Exception:
            self.fake_class_index = 1
        # Optional integrations (injected)
        self.telemetry = None
        self.active_learner = None
        # queue low-confidence below this to active learner
        self.queue_low_confidence_below = 0.05
        
        # Default thresholds for alert levels
        self.thresholds = confidence_thresholds or {
            'safe_max': 0.30,           # < 30% = SAFE
            'warning_max': 0.70,        # 30-70% = WARNING
            'danger_max': 0.95,         # 70-95% = DANGER
            'critical_min': 0.95        # > 95% = CRITICAL
        }
    
    def process_ensemble_output(
        self,
        ensemble_logits: torch.Tensor,
        individual_logits: list,
        frame_scores: torch.Tensor,
        video_id: str,
        uncertainty: float = 0.0
    ) -> EnsemblePrediction:
        """
        Process ensemble output and make decision
        
        Args:
            ensemble_logits: (num_classes,) or (1, num_classes) ensemble prediction
            individual_logits: List of (num_classes,) logits from each model
            frame_scores: (num_frames,) or (1, num_frames) confidence per frame
            video_id: Video identifier
            uncertainty: Uncertainty estimate (0-1, higher = more uncertain)
        
        Returns:
            EnsemblePrediction with decision and explanation
        """
        
        # Ensure proper shapes
        if ensemble_logits.dim() == 1:
            ensemble_logits = ensemble_logits.unsqueeze(0)
        if frame_scores.dim() > 1:
            frame_scores = frame_scores.squeeze()
        
        # Apply temperature scaling for calibration
        calibrated_logits = ensemble_logits / self.temperature
        ensemble_probs = F.softmax(calibrated_logits, dim=1)[0]  # (num_classes,)
        
        # Get individual probabilities
        individual_probs = []
        for logits in individual_logits:
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probs = F.softmax(logits / self.temperature, dim=1)[0]
            try:
                idx = int(self.fake_class_index)
            except Exception:
                idx = 1
            idx = 1 if idx not in (0, 1) else idx
            individual_probs.append(probs[idx].item())  # Fake class probability

        # Main prediction components
        try:
            idx = int(self.fake_class_index)
        except Exception:
            idx = 1
        idx = 1 if idx not in (0, 1) else idx
        fake_prob = ensemble_probs[idx].item()  # Probability of being fake

        # Agreement between individual models (1 - stddev)
        try:
            ind_arr = np.array(individual_probs, dtype=float)
            agreement = float(1.0 - np.std(ind_arr))
            mean_individual = float(np.mean(ind_arr))
        except Exception:
            agreement = 1.0
            mean_individual = fake_prob

        # Combine ensemble and individual mean, then apply uncertainty penalty
        adjusted_prob = (0.7 * fake_prob + 0.3 * mean_individual) * (
            1.0 - self.uncertainty_penalty * uncertainty
        )

        # Abstain logic: when uncertainty is high and models disagree, defer decision
        if (
            self.abstain_on_high_uncertainty
            and uncertainty > self.abstain_uncertainty_threshold
            and agreement < self.min_agreement_to_act
        ):
            is_fake = None
            confidence = max(0.0, (1.0 - uncertainty) * agreement)
            alert_level = AlertLevel.WARNING
            explanation = (
                f"Abstained: high uncertainty ({uncertainty:.2f}) and low model agreement ({agreement:.2f})."
            )

            # Telemetry + active-learning hook
            try:
                if self.telemetry:
                    self.telemetry.log_event({
                        'event': 'abstain',
                        'video_id': video_id,
                        'ensemble_prob': adjusted_prob,
                        'confidence': confidence,
                        'uncertainty': uncertainty,
                    })
            except Exception:
                pass

            try:
                if self.active_learner:
                    self.active_learner.queue_for_label({
                        'video_id': video_id,
                        'ensemble_prob': adjusted_prob,
                        'confidence': confidence,
                        'uncertainty': uncertainty,
                    })
            except Exception:
                pass

            return EnsemblePrediction(
                video_id=video_id,
                is_fake=is_fake,
                confidence=confidence,
                alert_level=alert_level,
                ensemble_prob=adjusted_prob,
                individual_probs=individual_probs,
                frame_scores=(frame_scores.cpu().numpy() if hasattr(frame_scores, 'cpu') else frame_scores),
                uncertainty=uncertainty,
                explanation=explanation,
            )

        # Determine binary decision from adjusted probability
        is_fake = adjusted_prob > self.decision_threshold

        # Apply uncertainty and agreement to compute final confidence
        confidence = abs(adjusted_prob - self.decision_threshold) * 2.0  # 0-1 scale
        confidence = confidence * max(0.0, agreement) * (1.0 - self.uncertainty_penalty * uncertainty)
        
        # Determine alert level
        alert_level = self._determine_alert_level(adjusted_prob, confidence, uncertainty)
        
        # Generate explanation
        explanation = self._generate_explanation(
            fake_prob, confidence, uncertainty, alert_level, individual_probs
        )

        # Telemetry: record decision
        try:
            if self.telemetry:
                self.telemetry.log_event({
                    'event': 'decision',
                    'video_id': video_id,
                    'is_fake': bool(is_fake),
                    'ensemble_prob': adjusted_prob,
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                    'alert_level': alert_level.name,
                })
        except Exception:
            pass

        # Active learning hook for very low confidence
        try:
            if self.active_learner and confidence < self.queue_low_confidence_below:
                self.active_learner.queue_for_label({
                    'video_id': video_id,
                    'ensemble_prob': adjusted_prob,
                    'confidence': confidence,
                    'uncertainty': uncertainty,
                })
        except Exception:
            pass
        
        return EnsemblePrediction(
            video_id=video_id,
            is_fake=is_fake,
            confidence=confidence,
            alert_level=alert_level,
            ensemble_prob=fake_prob,
            individual_probs=individual_probs,
            frame_scores=frame_scores.cpu().numpy() if hasattr(frame_scores, 'cpu') else frame_scores,
            uncertainty=uncertainty,
            explanation=explanation
        )
    
    def _determine_alert_level(
        self,
        fake_prob: float,
        confidence: float,
        uncertainty: float
    ) -> AlertLevel:
        """Determine alert level based on probability and uncertainty"""
        
        # Adjust thresholds based on uncertainty
        uncertainty_factor = 1.0 - 0.2 * uncertainty
        
        safe_max = self.thresholds['safe_max'] * uncertainty_factor
        warning_max = self.thresholds['warning_max'] * uncertainty_factor
        danger_max = self.thresholds['danger_max'] * uncertainty_factor
        
        if fake_prob < safe_max:
            return AlertLevel.SAFE
        elif fake_prob < warning_max:
            return AlertLevel.WARNING
        elif fake_prob < danger_max:
            return AlertLevel.DANGER
        else:
            return AlertLevel.CRITICAL
    
    def _generate_explanation(
        self,
        fake_prob: float,
        confidence: float,
        uncertainty: float,
        alert_level: AlertLevel,
        individual_probs: list
    ) -> str:
        """Generate human-readable explanation"""
        
        alert_names = {
            AlertLevel.SAFE: "AUTHENTIC",
            AlertLevel.WARNING: "UNCERTAIN",
            AlertLevel.DANGER: "LIKELY DEEPFAKE",
            AlertLevel.CRITICAL: "VERY LIKELY DEEPFAKE"
        }
        
        parts = [
            f"Classification: {alert_names[alert_level]}",
            f"Fake probability: {fake_prob*100:.1f}%",
            f"Confidence: {confidence*100:.1f}%"
        ]
        
        if uncertainty > 0.5:
            parts.append(f"High uncertainty detected ({uncertainty*100:.1f}%)")
        
        if individual_probs:
            avg_individual = np.mean(individual_probs)
            disagreement = np.std(individual_probs)
            parts.append(f"Model agreement: {(1-disagreement)*100:.1f}%")

        # If we abstained (is_fake can be None) provide guidance
        if confidence < 0.05 and uncertainty > 0.5:
            parts.append("Action: Abstain and request human review or collect more data")
        
        return " | ".join(parts)
    
    def batch_process(
        self,
        ensemble_logits: torch.Tensor,
        individual_logits_list: list,
        frame_scores: torch.Tensor,
        video_ids: list,
        uncertainties: Optional[torch.Tensor] = None
    ) -> list:
        """Process batch of ensemble predictions"""
        
        batch_size = ensemble_logits.shape[0]
        results = []
        
        for i in range(batch_size):
            uncertainty = uncertainties[i].item() if uncertainties is not None else 0.0
            
            individual_logits = [logits[i] for logits in individual_logits_list]
            
            result = self.process_ensemble_output(
                ensemble_logits=ensemble_logits[i],
                individual_logits=individual_logits,
                frame_scores=frame_scores[i] if frame_scores.dim() > 1 else frame_scores,
                video_id=video_ids[i] if isinstance(video_ids, list) else f"video_{i}",
                uncertainty=uncertainty
            )
            
            results.append(result)
        
        return results


class DecisionAggregator:
    """Aggregates multiple predictions and makes coordinated decisions"""
    
    def __init__(self):
        self.decision_history = []
    
    def aggregate_predictions(
        self,
        predictions: list,
        strategy: str = 'confidence_weighted'
    ) -> Dict:
        """
        Aggregate multiple predictions
        
        Args:
            predictions: List of EnsemblePrediction objects
            strategy: Aggregation strategy ('confidence_weighted', 'majority_voting', 'unanimous')
        
        Returns:
            Aggregated decision summary
        """
        
        if strategy == 'confidence_weighted':
            return self._aggregate_weighted(predictions)
        elif strategy == 'majority_voting':
            return self._aggregate_voting(predictions)
        elif strategy == 'unanimous':
            return self._aggregate_unanimous(predictions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _aggregate_weighted(self, predictions: list) -> Dict:
        """Confidence-weighted aggregation"""
        
        total_confidence = sum(p.confidence for p in predictions)
        if total_confidence == 0:
            total_confidence = len(predictions)
        
        weighted_prob = sum(
            p.ensemble_prob * p.confidence for p in predictions
        ) / total_confidence
        
        weighted_is_fake = weighted_prob > 0.5
        
        uncertainty = np.mean([p.uncertainty for p in predictions])
        
        return {
            'weighted_prob': weighted_prob,
            'is_fake': weighted_is_fake,
            'uncertainty': uncertainty,
            'num_predictions': len(predictions),
            'avg_confidence': total_confidence / len(predictions)
        }
    
    def _aggregate_voting(self, predictions: list) -> Dict:
        """Majority voting aggregation"""
        
        votes = sum(1 for p in predictions if p.is_fake)
        total = len(predictions)
        
        return {
            'fake_votes': votes,
            'total_votes': total,
            'is_fake': votes > total / 2,
            'agreement': votes / total if total > 0 else 0.5
        }
    
    def _aggregate_unanimous(self, predictions: list) -> Dict:
        """Unanimous decision (highest confidence only if all agree)"""
        
        all_fake = all(p.is_fake for p in predictions)
        all_authentic = all(not p.is_fake for p in predictions)
        
        if all_fake:
            decision = True
            confidence_level = 'HIGH'
        elif all_authentic:
            decision = False
            confidence_level = 'HIGH'
        else:
            # Disagreement
            decision = np.mean([p.ensemble_prob for p in predictions]) > 0.5
            confidence_level = 'LOW'
        
        return {
            'is_fake': decision,
            'confidence_level': confidence_level,
            'unanimity': all_fake or all_authentic,
            'num_predictions': len(predictions)
        }
