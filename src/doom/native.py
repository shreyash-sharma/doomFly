"""Native SIMD implementation of the fixed-step model; no changed graph."""
import ctypes as C
from pathlib import Path
import math,time,sys,os,json,hashlib
import numpy as np
from doom.engine import Brain
ROOT=Path(__file__).resolve().parents[1]
_default_library = 'libneural.dylib' if sys.platform == 'darwin' else ('libneural.dll' if sys.platform.startswith('win') else 'libneural.so')
LIBRARY=Path(os.environ.get('DOOM_KERNEL_PATH',str(ROOT/'outputs/doom'/_default_library)))
BUILD=json.loads(LIBRARY.with_suffix(LIBRARY.suffix+'.json').read_text())
if BUILD['kernel_source_sha256']!=hashlib.sha256((ROOT/'doom/kernel.cpp').read_bytes()).hexdigest() or BUILD['binary_sha256']!=hashlib.sha256(LIBRARY.read_bytes()).hexdigest():
    raise RuntimeError('Native model differs from reviewed source. Run python -m doom.build_kernel.')
_lib=C.CDLL(str(LIBRARY))
_f=_lib.neural_advance
_f.argtypes=[C.c_int]+[C.c_void_p]*11+[C.c_int,C.c_float]+[C.c_void_p]*5
_f.restype=None
class NativeBrain(Brain):
    def __init__(self,path,dt=.1):
        super().__init__(path,dt)
        self.previous_drive=np.zeros(self.n,dtype=np.float32)
        self.last=np.full(self.n,-1,dtype=np.int64)
    def step(self,luminance,duration_ms,sugar=False,lamina_bias=12.0,extra_drive=None):
        if len(luminance)!=len(self.retina) or not np.all(np.isfinite(luminance)):raise ValueError('Invalid retinal input')
        if not math.isfinite(duration_ms) or not math.isfinite(lamina_bias):raise ValueError('Finite duration and current required')
        steps=int(round(duration_ms/self.dt))
        if steps<1:raise ValueError('Duration too short')
        self.luminance+=(1-math.exp(-steps*self.dt/10))*(np.clip(luminance,0,1)-self.luminance)
        self.drive.fill(0);self.drive[self.lamina]=lamina_bias;self.drive[self.retina]=30*self.luminance/(.02+self.luminance)
        if sugar:self.drive[self.sugar]=30
        if extra_drive is not None:
            extra_drive=np.asarray(extra_drive,dtype=np.float32)
            if extra_drive.shape != (self.n,) or not np.all(np.isfinite(extra_drive)):
                raise ValueError('Extra drive must be a finite vector with one value per neuron')
            self.drive += extra_drive
            newly_active=np.flatnonzero((extra_drive != 0) & (self.active_flag == 0))
            if len(newly_active):
                start=int(self.nactive[0]); stop=start+len(newly_active)
                self.active[start:stop]=newly_active; self.active_flag[newly_active]=1; self.nactive[0]=stop
        self.counts.fill(0);clock=np.asarray([self.cursor],dtype=np.int64)
        arrays=[self.ptr,self.post,self.weight,self.v,self.g,self.refractory,self.drive,self.previous_drive,self.queue,self.queue_count,clock]
        start=time.perf_counter()
        _f(self.n,*[x.ctypes.data for x in arrays],steps,self.dt,*[x.ctypes.data for x in [self.counts,self.active,self.active_flag,self.nactive,self.last]])
        wall=time.perf_counter()-start
        self.cursor=int(clock[0]);self.total_spikes+=int(self.counts.sum());self.sim_ms+=steps*self.dt
        return self.counts.copy(),wall
