"""Public reproducibility metadata; never includes environment variables."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def provenance(graph,build,assets):
    audit=json.loads((ROOT/'outputs/doom/audit/data-integrity.json').read_text())
    actual=sha(graph)
    if not audit['passed'] or actual!=audit['graph_sha256']:
        raise RuntimeError('Runtime graph does not match the independently audited graph.')
    files=sorted((ROOT/'doom').glob('*.py'))+sorted((ROOT/'doom').glob('*.cpp'))
    return {'model_revision':build['model_revision'],'kernel':build,'graph_sha256':actual,
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'assets':assets,
      'data_audit_sha256':sha(ROOT/'outputs/doom/audit/data-integrity.json')}
