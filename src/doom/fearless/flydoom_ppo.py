"""Clipped PPO for the small frozen-MaleCNS linear action readout."""
from __future__ import annotations
import numpy as np

class ClippedPPO:
    def __init__(self, actor, value, *, clip=.2, actor_lr=3e-3, value_lr=1e-2,
                 epochs=4, batch_size=64, gamma=.99, gae_lambda=.95,
                 max_grad_norm=.5, seed=20260917):
        self.actor,self.value=actor,value;self.clip=clip;self.actor_lr=actor_lr;self.value_lr=value_lr
        self.epochs,self.batch_size=epochs,batch_size;self.gamma,self.gae_lambda=gamma,gae_lambda
        self.max_grad_norm=max_grad_norm;self.rng=np.random.default_rng(seed)
    @staticmethod
    def _softmax(z):
        q=np.exp(np.clip(z-np.max(z),-30,30));return q/q.sum()
    def update(self,traj,gamma=None):
        if not traj:return {'policy_loss':0.,'value_loss':0.,'clip_fraction':0.,'gradient_norm':0.}
        gamma=self.gamma if gamma is None else gamma;n=len(traj)
        rewards=np.asarray([t['reward'] for t in traj]);oldv=np.asarray([t['value'] for t in traj])
        oldlog=np.log(np.asarray([t['probability'][t['index']] for t in traj]).clip(1e-8))
        adv=np.zeros(n);gae=0.;nextv=0.
        for i in range(n-1,-1,-1):
            delta=rewards[i]+gamma*nextv-oldv[i];gae=delta+gamma*self.gae_lambda*gae;adv[i]=gae;nextv=oldv[i]
        returns=adv+oldv;adv=(adv-adv.mean())/(adv.std()+1e-8);pls=[];vls=[];clips=[];norms=[]
        for _ in range(self.epochs):
            order=self.rng.permutation(n)
            for start in range(0,n,self.batch_size):
                batch=order[start:start+self.batch_size];ga=np.zeros_like(self.actor);gv=np.zeros_like(self.value)
                for i in batch:
                    t=traj[i];x=t['vector'];p=self._softmax(self.actor@x);ratio=np.exp(np.log(max(p[t['index']],1e-8))-oldlog[i])
                    raw=ratio*adv[i];bounded=np.clip(ratio,1-self.clip,1+self.clip)*adv[i]
                    pls.append(-min(raw,bounded));clips.append(abs(ratio-1)>self.clip)
                    if raw<=bounded:
                        one=np.zeros(len(p));one[t['index']]=1;ga+=-adv[i]*ratio*np.outer(one-p,x)
                    error=float(self.value@x)-returns[i];gv+=2*error*x;vls.append(error*error)
                scale=max(1,len(batch));ga/=scale;gv/=scale;norm=float(np.sqrt(np.sum(ga*ga)+np.sum(gv*gv)))
                if norm>self.max_grad_norm:
                    factor=self.max_grad_norm/norm;ga*=factor;gv*=factor
                self.actor-=self.actor_lr*ga;self.value-=self.value_lr*gv;norms.append(norm)
        return {'policy_loss':float(np.mean(pls)),'value_loss':float(np.mean(vls)),
                'clip_fraction':float(np.mean(clips)),'gradient_norm':float(np.mean(norms))}
