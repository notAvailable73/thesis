"""TMLR plan Task 1 — ResNet-34 and MobileNetV2 frozen backbones (E2).

Offline: every backbone is built with pretrained=False (no weight download).
"""
from __future__ import annotations

import torch

from src.adapters import default_lora_targets, resolve_stage_paths
from src.backbones import backbone_feature_dim, build_backbone


def test_resnet34_contract():
    bb = build_backbone("resnet34", pretrained=False)
    with torch.no_grad():
        assert bb(torch.zeros(2, 3, 224, 224)).shape == (2, 512)
    assert not any(p.requires_grad for p in bb.parameters())
    assert not bb.training
    assert backbone_feature_dim("resnet_34") == 512


def test_mobilenetv2_contract():
    bb = build_backbone("mbnet2", pretrained=False)
    with torch.no_grad():
        assert bb(torch.zeros(2, 3, 224, 224)).shape == (2, 1280)
    assert not any(p.requires_grad for p in bb.parameters())
    assert not bb.training
    assert backbone_feature_dim("mobilenet_v2") == 1280


def test_stage_paths_new_backbones():
    assert resolve_stage_paths(build_backbone("resnet34", pretrained=False)) == [
        "layer1.2", "layer2.3", "layer3.5", "layer4.2"]
    assert resolve_stage_paths(build_backbone("mobilenetv2", pretrained=False)) == [
        "features.3", "features.6", "features.10", "features.13", "features.16"]


def test_default_lora_targets_new_backbones():
    assert default_lora_targets(build_backbone("resnet34", pretrained=False)) == [
        "layer4.0.downsample.0"]
    assert default_lora_targets(build_backbone("mobilenetv2", pretrained=False)) == [
        "features.17.conv.2"]
