"""Test-session setup: torch and LightGBM ship separate OpenMP runtimes on macOS and can segfault together."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
