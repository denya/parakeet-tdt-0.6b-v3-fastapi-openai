"""Run the unmodified service with optional diagnostic instrumentation."""
import atexit
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import runpy
import signal
import subprocess
import sys
import threading
import time

stats = {}
lock = threading.Lock()

def timed(name, fn):
    def call(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            with lock:
                item = stats.setdefault(name, {"seconds": 0.0, "calls": 0})
                item["seconds"] += time.perf_counter() - start
                item["calls"] += 1
    return call

if os.environ.get("BENCH_PROFILE") == "1":
    import onnxruntime as ort
    from onnx_asr.adapters import AsrAdapter
    AsrAdapter.recognize = timed("recognize_total", AsrAdapter.recognize)
    original_run = ort.InferenceSession.run
    def ort_run(self, *args, **kwargs):
        name = Path(str(getattr(self, "_model_path", "preprocessor"))).name
        return timed("onnx:" + name, original_run)(self, *args, **kwargs)
    ort.InferenceSession.run = ort_run
    original_subprocess = subprocess.run
    def child_run(args, *rest, **kwargs):
        return timed(Path(args[0]).name, original_subprocess)(args, *rest, **kwargs)
    subprocess.run = child_run

@atexit.register
def save():
    target = os.environ.get("BENCH_WORKER_REPORT")
    if target:
        Path(target).write_text(json.dumps({
            "python": sys.version,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "packages": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            "profile": stats,
        }, indent=2))

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
runpy.run_path(sys.argv[1], run_name="__main__")
