"""Build atomically; bind a native binary to its exact reviewed source."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path)
    args=p.parse_args()
    default = 'libneural.dylib' if sys.platform == 'darwin' else ('libneural.dll' if sys.platform.startswith('win') else 'libneural.so')
    out=args.output or ROOT/'outputs/doom'/default
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix(out.suffix+'.partial')
    source=ROOT/'doom/kernel.cpp'
    command=['clang++','-O3','-std=c++17','-shared']
    if not sys.platform.startswith('win'): command.append('-fPIC')
    command += [str(source),'-o',str(temporary)]
    subprocess.run(command,check=True)
    record={'kernel_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'binary_sha256':hashlib.sha256(temporary.read_bytes()).hexdigest(),
      'model_revision':'lif-r2-refractory-write-protection','compile_flags':command[1:5]}
    temporary.replace(out)
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
if __name__=='__main__':main()
