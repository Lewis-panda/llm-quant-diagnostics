"""Calibration data.

AWQ 原版用 128 段 Pile 樣本做 calibration；診斷用途只需要少量、主題多元的
文字就足以觀察 activation 分布。These few paragraphs span ML / math / general
prose so the collected per-channel statistics are not dominated by one register.
"""
from __future__ import annotations

from typing import List

CALIBRATION_TEXTS: List[str] = [
    "The transformer architecture was introduced in the paper Attention Is All You Need. "
    "It relies entirely on self-attention mechanisms to draw global dependencies between "
    "input and output. The key innovation was replacing recurrent layers with multi-head "
    "attention, allowing for significantly more parallelization during training.",

    "In mathematics, a matrix is a rectangular array of numbers arranged in rows and columns. "
    "The individual items in a matrix are called its elements or entries. Matrices have wide "
    "applications in engineering, physics, economics, and statistics. The determinant of a "
    "square matrix is a scalar value that encodes certain properties of the linear transformation.",

    "Machine learning is a subset of artificial intelligence that provides systems the ability "
    "to automatically learn and improve from experience without being explicitly programmed. "
    "The process begins with observations or data, such as examples, direct experience, or "
    "instruction, in order to look for patterns in data and make better decisions.",

    "Quantization reduces the precision of weights and activations in neural networks from "
    "floating point to lower bit-width integers. This compression technique significantly "
    "reduces memory footprint and enables faster inference on hardware that supports integer "
    "arithmetic. However, aggressive quantization can lead to accuracy degradation.",
]


def get_calibration_texts() -> List[str]:
    return list(CALIBRATION_TEXTS)
