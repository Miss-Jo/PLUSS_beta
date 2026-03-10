"""
PLUSS_β: Point and Box Labeling with Unified Self-Supervised System (Beta Version)

A self-supervised framework for language-driven unsupervised semantic segmentation
that enhances the PLUSS pipeline through trainable semantic and box tuners.

Key Components:
- Semantic Tuner: Visual prompt tuning for improved semantic discrimination
- Box Tuner: Region-level feature fusion for accurate spatial localization  
- Memory Bank: Hard example storage for targeted learning
- Point-2-Box: Conversion of point prompts to pseudo bounding boxes

Design Principle:
The semantic tuner and box tuner are trained with COMPLETELY SEPARATE computational
graphs and gradient paths. This architectural isolation ensures:
1. Semantic Tuner specializes in improving semantic discrimination
2. Box Tuner specializes in improving spatial localization
3. No interference between the two learning objectives
"""

__version__ = '1.0.0'

from pluss_beta.models.semantic_tuner import SemanticTuner, SemanticTunerLoss
from pluss_beta.models.box_tuner import BoxTuner, BoxTunerLoss
from pluss_beta.models.memory_bank import MemoryBank
from pluss_beta.utils.point2box import Point2BoxConverter, point_to_box
from pluss_beta.trainer import PLUSSBetaTrainer
from pluss_beta.inference import PLUSSBetaInference, load_trained_model

__all__ = [
    'SemanticTuner',
    'SemanticTunerLoss',
    'BoxTuner',
    'BoxTunerLoss',
    'MemoryBank',
    'Point2BoxConverter',
    'point_to_box',
    'PLUSSBetaTrainer',
    'PLUSSBetaInference',
    'load_trained_model'
]
