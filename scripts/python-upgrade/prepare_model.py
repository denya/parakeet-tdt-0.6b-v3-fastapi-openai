"""Download only the pinned INT8 model files into a dedicated test cache."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = '8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce'
FILES = ['config.json','vocab.txt','encoder-model.int8.onnx','decoder_joint-model.int8.onnx']
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
path=Path(snapshot_download('istupakov/parakeet-tdt-0.6b-v3-onnx',revision=REVISION,cache_dir=a.cache,allow_patterns=FILES))
# The application's loader resolves main while offline. Point ONLY this dedicated test cache at the pinned snapshot.
refs=path.parent.parent/'refs';refs.mkdir(exist_ok=True);(refs/'main').write_text(REVISION)
print(json.dumps({'revision':REVISION,'files':{f:hashlib.sha256((path/f).read_bytes()).hexdigest() for f in FILES}},indent=2))
