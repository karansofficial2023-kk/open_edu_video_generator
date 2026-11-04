"""Launcher for this machine's portable Python; does not modify ComfyUI."""
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path[:0] = [str(root), str(root / '.runtime-deps')]
runpy.run_module('app.main', run_name='__main__')
