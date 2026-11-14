"""CI sets D3GS_REQUIRE_TORCH=1 so the torch-dependent tests can never be skipped there unnoticed."""
import os

if os.environ.get("D3GS_REQUIRE_TORCH") == "1":
    import torch  # noqa: F401  (ImportError here fails the whole run)
