"""TriGuard ML package."""
import os

# torch and LightGBM ship separate OpenMP runtimes on macOS; together they can segfault.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
