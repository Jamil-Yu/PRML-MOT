# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = '8.0.82'
import sys

import logging

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics.hub import start
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils.checks import check_yolo as checks

__all__ = '__version__', 'YOLO', 'checks', 'start'  # allow simpler import
