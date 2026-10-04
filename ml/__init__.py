"""TriGuard ML package."""
import os

# LightGBM and torch ship separate OpenMP runtimes on macOS. Safe combination found empirically:
# import LightGBM first, then torch, and keep torch single-threaded (otherwise torch ops hang or
# LightGBM unpickling segfaults). This must run before anything else imports either library.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import lightgbm  # noqa: F401
except ImportError:  # optional for the rules/TF-IDF-only paths
    pass
try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass
