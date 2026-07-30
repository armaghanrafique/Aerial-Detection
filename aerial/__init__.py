"""Aerial object detection for drone / quadcopter footage.

A thin, CPU-friendly layer over Ultralytics YOLO with optional SAHI slicing
(for tiny top-down objects) and a dependency-free tracker. See `cli.py`.
"""
from .detector import AerialDetector, COCO_OUTDOOR
from .tracker import SimpleTracker

__all__ = ["AerialDetector", "SimpleTracker", "COCO_OUTDOOR"]
