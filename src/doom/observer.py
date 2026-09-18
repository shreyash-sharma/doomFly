"""Read-only native Doom spectator; a checked mirror never feeds the brain.

Only the primary game's actions advance this engine. Camera calls render its
current state without a tic. Every copied action/reset is checked against the
primary's RGB, game variables and object state. Divergence disables this view.
"""
import base64,hashlib,io,json,math,os,struct,tempfile,threading,time
from pathlib import Path
import numpy as np
from PIL import Image
from doom.game import Game

ROOT=Path(__file__).resolve().parents[1]
ENGINE=ROOT/'outputs/doom/native-spectator-v1/engine/vizdoom'

class ObserverUnavailable(RuntimeError):pass
class ObserverBusy(RuntimeError):pass

def camera_query(query):
    """Fixed numeric protocol, never console commands, paths or model inputs."""
    from urllib.parse import parse_qs
    if len(query)>256:raise ValueError('Invalid camera')
    values=parse_qs(query,strict_parsing=True)
    if set(values)!={'x','y','z','yaw','pitch'} or any(len(v)!=1 for v in values.values()):raise ValueError('Invalid camera')
    result=[float(values[k][0]) for k in ['x','y','z','yaw','pitch']]
    if not all(math.isfinite(v) for v in result):raise ValueError('Invalid camera')
    x,y,z,yaw,pitch=result
    if abs(x)>8192 or abs(y)>8192 or not 4<=z<=2048 or not 0<=yaw<360 or abs(pitch)>60:raise ValueError('Camera outside range')
    return result

def _image(image,fmt='PNG'):
    f=io.BytesIO();image.save(f,format=fmt,**({'quality':87} if fmt=='JPEG' else {}))
    return 'data:image/'+('jpeg' if fmt=='JPEG' else 'png')+';base64,'+base64.b64encode(f.getvalue()).decode()

class NativeObserver:
    def __init__(self,primary,seed,scenario,engine=ENGINE):
        self.lock=threading.Lock();self.ready=False;self.error=None;self.nonce=0
        self.last_render=0.;self.last_verified=0.;self.verified_ticks=0;self.last_packet=None
        self.cache={};self.closed=False
        if not Path(engine).is_file():raise ObserverUnavailable('Native observer is not installed')
        self.directory=tempfile.TemporaryDirectory(prefix='doomfly-observer-')
        self.output=Path(self.directory.name)/'view.bin'
        previous=os.environ.get('DOOMFLY_OBSERVER_OUTPUT')
        os.environ['DOOMFLY_OBSERVER_OUTPUT']=str(self.output)
        try:self.mirror=Game(seed=seed,scenario=scenario,spectator=True,observer_engine=engine)
        finally:
            if previous is None:os.environ.pop('DOOMFLY_OBSERVER_OUTPUT',None)
            else:os.environ['DOOMFLY_OBSERVER_OUTPUT']=previous
        self.mirror.episode=primary.episode
        self.fingerprint={'engine_sha256':hashlib.sha256(Path(engine).read_bytes()).hexdigest(),
            'resource_sha256':hashlib.sha256(Path(engine).with_name('vizdoom.pk3').read_bytes()).hexdigest()}
        self._verify(primary)

    def _verify(self,primary):
        self.ready=False
        if primary.observation()!=self.mirror.observation():raise ObserverUnavailable('Native observer game state diverged')
        if not primary.observation()['finished']:
            if not np.array_equal(primary.pixels(),self.mirror.pixels()):raise ObserverUnavailable('Native observer RGB diverged')
            if primary.spectator()!=self.mirror.spectator():raise ObserverUnavailable('Native observer objects diverged')
            self.ready=True
        self.verified_ticks+=1;self.last_verified=time.monotonic()
        self.game_state=primary.observation();self.pose=primary.spectator()

    def advance(self,primary,action=None,reset=False):
        if self.error or self.closed:return
        with self.lock:
            try:
                if reset:self.mirror.new_episode()
                else:self.mirror.act(action)
                self._verify(primary)
            except Exception as e:
                self.ready=False;self.error=str(e)
                # Observer faults never stop, reset or modify the primary game.
                import logging
                logging.getLogger('doom-observer').exception('Native observer disabled')

    def render(self,pose):
        if not self.lock.acquire(blocking=False):raise ObserverBusy('Observer busy')
        try:
            now=time.monotonic()
            if not self.ready or self.closed or self.error or now-self.last_verified>5:raise ObserverUnavailable('No verified live observer frame')
            if now-self.last_render<1/24:raise ObserverBusy('Observer capacity')
            self.last_render=now
            # Keep camera inside the convex four-wall arena. It cannot fly into
            # void space where the Doom renderer has no valid geometry.
            points=[p for s in self.pose['sectors'] for l in s['lines'] for p in [(l[0],l[1]),(l[2],l[3])]]
            x0,x1=min(p[0] for p in points)+4,max(p[0] for p in points)-4
            y0,y1=min(p[1] for p in points)+4,max(p[1] for p in points)-4
            pose=[max(x0,min(x1,pose[0])),max(y0,min(y1,pose[1])),*pose[2:]]
            self.nonce=(self.nonce%2147483646)+1
            self.mirror.game.send_game_command('doomfly_view '+str(self.nonce)+' '+' '.join(f'{v:.5f}' for v in pose))
            deadline=time.monotonic()+.35
            raw=None;meta=None
            while time.monotonic()<deadline:
                if self.output.is_file():
                    candidate=self.output.read_bytes()
                    if candidate[:8]!=b'DFVIEW01':raise ObserverUnavailable('Invalid observer protocol')
                    size=struct.unpack('<I',candidate[8:12])[0]
                    if size>16000:raise ObserverUnavailable('Invalid observer metadata')
                    metadata=json.loads(candidate[12:12+size])
                    if metadata['nonce']==self.nonce:raw=candidate;meta=metadata;break
                time.sleep(.001)
            if raw is None:raise ObserverUnavailable('Observer render timed out')
            w,h=meta['width'],meta['height']
            if (w,h)!=(640,480):raise ObserverUnavailable('Unexpected observer resolution')
            offset=12+size;rgb=raw[offset:offset+w*h*3];offset+=w*h*3
            depth=raw[offset:offset+w*h];offset+=w*h
            for layer in meta['weapon']:
                n=layer['width']*layer['height']*4;pixels=raw[offset:offset+n];offset+=n
                digest=hashlib.sha256(pixels).hexdigest()
                if digest not in self.cache:
                    if len(self.cache)>64:self.cache.clear()
                    self.cache[digest]=_image(Image.frombytes('RGBA',(layer['width'],layer['height']),pixels))
                layer['image']=self.cache[digest]
            if offset!=len(raw):raise ObserverUnavailable('Invalid observer buffer size')
            meta.update({'image':_image(Image.frombytes('RGB',(w,h),rgb),'JPEG'),
                'depth':_image(Image.frombytes('L',(w,h),depth)),
                'game':self.game_state,'player':self.pose['player'],
                'generated_at_ms':int(time.time()*1000),'verified_ticks':self.verified_ticks})
            return json.dumps(meta,separators=(',',':')).encode()
        finally:self.lock.release()

    def close(self):
        with self.lock:
            self.closed=True;self.ready=False;self.mirror.close();self.directory.cleanup()
