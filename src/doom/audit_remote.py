"""Compare local raw source bytes with the official GCS object's MD5."""
import base64,hashlib,json,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/'connectome_data/malecns_v1';lock=json.loads((root/'source.lock.json').read_text());results={}
    for name,info in lock.items():
        assert info['url'].startswith('https://storage.googleapis.com/flyem-male-cns/v1.0/')
        with urllib.request.urlopen(urllib.request.Request(info['url'],method='HEAD'),timeout=30) as r:
            remote=','.join(r.headers.get_all('x-goog-hash',[]))
            md5=next(x.strip()[4:] for x in remote.split(',') if x.strip().startswith('md5='))
            size=int(r.headers['Content-Length'])
        h=hashlib.md5()
        with (root/name).open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        assert base64.b64encode(h.digest()).decode()==md5 and size==(root/name).stat().st_size
        results[name]={'url':info['url'],'remote_md5_base64':md5,'bytes':size,'matched':True}
    (ROOT/'outputs/doom/audit/official-source-check.json').write_text(json.dumps(results,indent=2)+'\n')
    print('All three local files match the official GCS object MD5 and size.')
if __name__=='__main__':main()
