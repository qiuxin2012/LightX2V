import os


# Benchmark-only switch.  Keep this module deliberately small so callers can
# use the constant from torch.compile'd paths without introducing graph breaks.
SKIP_DISTRIBUTED_COMM = os.getenv("LIGHTX2V_SKIP_DISTRIBUTED_COMM", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
