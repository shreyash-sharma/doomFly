"""Small pixel-only visual encoder inspired by FlyDoom-style motion features.

It is an engineered sensory front end, not a claim about Drosophila retinal
physiology. It exposes coarse left/centre/right motion and looming energy for
translation into LC4/LPLC2 current; it never emits a Doom action.
"""
import numpy as np

class MotionLoomEncoder:
    def __init__(self, width=32, height=24):
        self.width=int(width);self.height=int(height);self.previous=None
    def _luma(self,rgb):
        image=np.asarray(rgb,dtype=np.float32)/255.
        y=image@np.asarray([.2126,.7152,.0722],dtype=np.float32)
        ys=np.linspace(0,y.shape[0]-1,self.height).astype(int);xs=np.linspace(0,y.shape[1]-1,self.width).astype(int)
        return y[np.ix_(ys,xs)]
    def encode(self,rgb):
        y=self._luma(rgb);delta=np.zeros_like(y) if self.previous is None else np.abs(y-self.previous);self.previous=y
        thirds=np.array([delta[:,:self.width//3].mean(),delta[:,self.width//3:2*self.width//3].mean(),delta[:,2*self.width//3:].mean()])
        # A centre-weighted expansion proxy catches rapidly growing central motion.
        cy,cx=self.height//2,self.width//2;central=delta[max(0,cy-5):cy+5,max(0,cx-7):cx+7].mean()
        looming=float(np.clip(central*8.+delta.mean()*4.,0.,1.))
        values=np.clip(np.r_[thirds*8.,looming],0.,1.)
        return {'left':float(values[0]),'center':float(values[1]),'right':float(values[2]),'looming':float(values[3]),'vector':values.tolist()}
