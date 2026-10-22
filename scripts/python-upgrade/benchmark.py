"""Local-only real HTTP/API regression and latency runner; no production endpoint.

Run with the interpreter whose dependencies you want to test. Requires requests/psutil.
Corpus manifest: [{id, path, text, source, license}]. Relative paths resolve at manifest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata

import psutil
import requests

ROOT = Path(__file__).resolve().parents[2]
KEY = "isolated-python-upgrade-test"

def normalize(s):
    return re.sub(r"[^\w\s]", "", unicodedata.normalize("NFKC", s).lower()).split()

def distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = new
    return row[-1]

def errors(ref, hyp):
    a, b = normalize(ref), normalize(hyp)
    ca, cb = " ".join(a), " ".join(b)
    return {"wer": distance(a,b)/max(1,len(a)), "cer": distance(ca,cb)/max(1,len(ca))}

def summary(values):
    s = sorted(values)
    return {"n":len(s), "p50_s":statistics.median(s), "p95_s":s[math.ceil(.95*len(s))-1]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=20)
    p.add_argument('--cpus',help='Comma-separated CPU IDs; default first four allowed CPUs')
    p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    if a.rounds < 1: p.error('rounds must be positive')
    os.sched_setaffinity(0,{int(x) for x in a.cpus.split(',')} if a.cpus else set(sorted(os.sched_getaffinity(0))[:4]))
    corpus=json.loads(a.manifest.read_text())
    for item in corpus:
        item['path']=str((a.manifest.parent/item['path']).resolve())
        item['sha256']=hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()
        item['duration_s']=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',item['path']],text=True))
    a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    result={'python':sys.version,'platform':platform.platform(),'cpu':platform.processor(),
            'affinity':sorted(os.sched_getaffinity(0)),'rounds':a.rounds,'profile_enabled':a.profile,
            'ort_threads':[4,1],'waitress_threads':4,'max_active':2,
            'model':'nemo-parakeet-tdt-0.6b-v3','quantization':'int8','providers':['CPUExecutionProvider'],
            'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
            'corpus':corpus,'checks':[],'samples':[]}
    with tempfile.TemporaryDirectory(prefix='parakeet-ab-') as td:
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        env=os.environ.copy()
        env.update(API_KEY=KEY,HOST='127.0.0.1',PORT=str(port),PARAKEET_BACKEND='onnx',
                   PARAKEET_OPEN_BROWSER='0',PARAKEET_LISTEN='',PARAKEET_MODEL_CACHE=str(a.cache.resolve()),
                   HF_HUB_OFFLINE='1',HF_HUB_DISABLE_XET='1',ONNX_INTRA_OP_THREADS='4',
                   ONNX_INTER_OP_THREADS='1',WAITRESS_THREADS='4',PARAKEET_MAX_ACTIVE_TRANSCRIPTIONS='2',
                   OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',BENCH_PROFILE=str(int(a.profile)),
                   BENCH_WORKER_REPORT=str(a.output)+'.worker.json')
        log=open(str(a.output)+'.log','w')
        start=time.perf_counter()
        child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('worker.py')),str(ROOT/'app.py')],cwd=td,env=env,stdout=log,stderr=subprocess.STDOUT)
        session=requests.Session();session.trust_env=False
        rss=[]
        try:
            while True:
                if child.poll() is not None: raise RuntimeError('Service exited: '+Path(log.name).read_text()[-3000:])
                try:
                    response=session.get(url+'/health',timeout=1)
                    if response.status_code==200:break
                except requests.RequestException: pass
                if time.perf_counter()-start>180:raise TimeoutError('startup')
                time.sleep(.05)
            result['startup_cached_model_s']=time.perf_counter()-start
            result['health']=response.json();assert response.json()['backend']=='onnx'
            def check(name, condition):
                result['checks'].append({'name':name,'ok':bool(condition)})
                assert condition,name
            check('missing auth',session.post(url+'/v1/audio/transcriptions').status_code==401)
            check('wrong auth',session.post(url+'/v1/audio/transcriptions',headers={'Authorization':'Bearer wrong'}).status_code==401)
            headers={'Authorization':'Bearer '+KEY}
            check('missing file',session.post(url+'/v1/audio/transcriptions',headers=headers).status_code==400)
            check('openapi',session.get(url+'/openapi.json').status_code==200)
            def transcribe(path,**params):
                with open(path,'rb') as f:
                    start=time.perf_counter()
                    r=session.post(url+'/v1/audio/transcriptions',headers=headers,files={'file':(Path(path).name,f)},data={'model':'whisper-1',**params},timeout=180)
                    elapsed=time.perf_counter()-start
                if r.status_code!=200: raise RuntimeError(f'{r.status_code}: {r.text[:1000]}')
                return r,elapsed
            # First request is reported separately; warmups excluded from percentile samples.
            r,elapsed=transcribe(corpus[0]['path'])
            result['first_request_s']=elapsed
            result['first_text']=r.json()['text']
            for item in corpus:transcribe(item['path'])
            for n in range(a.rounds):
                for item in corpus:
                    r,elapsed=transcribe(item['path'])
                    text=r.json()['text']
                    result['samples'].append({'id':item['id'],'iteration':n,'seconds':elapsed,'rtf':elapsed/item['duration_s'],'text':text,**errors(item['text'],text)})
            result['warm']=summary([x['seconds'] for x in result['samples']])
            result['by_audio']={item['id']:summary([x['seconds'] for x in result['samples'] if x['id']==item['id']]) for item in corpus}
            # API/codec correctness after latency runs.
            source=corpus[0]['path']
            expected=next(x['text'] for x in result['samples'] if x['id']==corpus[0]['id'])
            for fmt in ['json','text','verbose_json','srt','vtt']:
                r,_=transcribe(source,response_format=fmt,language='en',temperature='0',prompt='John Kennedy',**{'timestamp_granularities[]':'word'})
                if fmt in ['json','verbose_json']:check('format '+fmt,r.json()['text']==expected)
                elif fmt=='text':check('format text',r.text==expected)
                else:check('format '+fmt,'-->' in r.text and (fmt!='vtt' or r.text.startswith('WEBVTT')))
            for suffix,opts in [('flac',[]),('mp3',[]),('m4a',[]),('ogg',['-c:a','libopus']),('webm',['-c:a','libopus']),('wav',['-ac','2','-ar','44100'])]:
                path=str(Path(td)/('codec.'+suffix))
                subprocess.run(['ffmpeg','-v','error','-y','-i',source,*opts,path],check=True)
                r,_=transcribe(path)
                check('codec '+suffix,bool(r.json()['text']))
                result.setdefault('codec_texts',{})[suffix]=r.json()['text']
            r=session.post(url+'/v1/audio/transcriptions',headers=headers,files={'file':('invalid.wav',b'not audio')},timeout=20)
            check('invalid audio rejected',400<=r.status_code<600)
            # Long input exercises actual silence detection / chunking.
            long_path=str(Path(td)/'long.wav')
            subprocess.run(['ffmpeg','-v','error','-y','-stream_loop','10','-i',source,'-t','100',long_path],check=True)
            r,_=transcribe(long_path,response_format='verbose_json')
            check('long chunked audio',len(r.json()['segments'])>=2)
            def concurrent_request(_):
                r,t=transcribe(source);return {'seconds':t,'text':r.json()['text']}
            start=time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as pool:con=list(pool.map(concurrent_request,range(8)))
            elapsed=time.perf_counter()-start
            result['concurrency_2']={'requests':8,'elapsed_s':elapsed,'requests_per_s':8/elapsed,'audio_seconds_per_s':8*corpus[0]['duration_s']/elapsed,'latency':summary([x['seconds'] for x in con])}
            check('concurrent text stable',all(x['text']==expected for x in con))
            check('temporary files cleaned',not list((Path(td)/'temp_uploads').iterdir()))
        finally:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            log.close();session.close()
            worker=Path(str(a.output)+'.worker.json')
            if worker.exists():
                result['worker']=json.loads(worker.read_text())
                result['peak_rss_mib']=result['worker']['peak_rss_mib']
            a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['startup_cached_model_s','first_request_s','warm','peak_rss_mib']},indent=2))

if __name__=='__main__':main()
