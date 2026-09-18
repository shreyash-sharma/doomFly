"""Render a frozen DoomFly replay as a shareable gameplay/neural-activity GIF."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from doom.fearless.doom_guy_train import ActorCritic,groups,run_episode

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'outputs/fearless/doom-guy-fly-v5-strafe/doom-guy-decoder.npz'
OUT=ROOT/'outputs/fearless/doom-guy-fly-v5-longrun/doomfly-longrun-brain-hud-788001.gif'
SEED=788001
BRAIN_ART=ROOT/'outputs/fearless/assets/doomfly-brain-illustrative-v1.png'

# Stable illustrative layout: sampled real graph neurons placed inside a
# bilateral fly-brain silhouette. Positions are aesthetic, activity is real.
_layout_rng=np.random.default_rng(787001)
BRAIN_SAMPLE=np.sort(_layout_rng.choice(166700,520,replace=False))
NODE_POS=[]
for i in range(len(BRAIN_SAMPLE)):
    side=-1 if i%2==0 else 1
    while True:
        x,y=_layout_rng.uniform(-1,1,2)
        if x*x+y*y<=1:break
    NODE_POS.append((820+side*73+x*67,337+y*82))
NODE_POS=np.asarray(NODE_POS)
EDGES=[tuple(_layout_rng.choice(len(BRAIN_SAMPLE),2,replace=False)) for _ in range(180)]
ART=Image.open(BRAIN_ART).convert('RGBA').resize((330,165))

def render(tick,rgb,counts,features,action,macro,game,packet):
    game_img=Image.fromarray(rgb).resize((640,480))
    canvas=Image.new('RGB',(1000,480),(8,12,18));canvas.paste(game_img,(0,0));d=ImageDraw.Draw(canvas)
    d.rectangle((640,0,999,479),fill=(10,18,27));white=(235,242,248);cyan=(44,210,210);orange=(255,145,45)
    d.text((662,20),'DOOMFLY // MaleCNS',fill=cyan);d.text((662,52),f'Time  {tick/35:4.1f}s',fill=white)
    d.text((662,76),f'Health {game["health"]:3d}   Kills {game["kills"]}',fill=white)
    d.text((662,112),f'Action: {macro}',fill=orange)
    d.text((662,140),f'forward {action["forward"]:>4.0f}  strafe {action.get("strafe",0):>4.0f}',fill=white)
    d.text((662,164),f'turn {action["turn"]:>4.0f}  attack {int(action["attack"])}',fill=white)
    d.text((662,200),f'Threat: {packet["direction"]}  {packet["intensity"]:.2f}',fill=cyan)
    d.text((662,228),'DoomFly neural activity',fill=white)
    canvas.paste(ART,(655,248),ART)
    # These population pulses are real replay activity, over an explicitly
    # illustrative brain image used for communication rather than anatomy.
    spots={'LC4-L':(725,294),'LC4-R':(917,294),'LPLC2-L':(748,351),'LPLC2-R':(892,351),'Tk-FruM':(820,382)}
    for name,ix in groups()[-6:-1]:
        x,y=spots[name];value=float(np.mean(counts[ix]));r=5+min(18,int(value*2))
        # translucent-looking concentric rings make changing real activity
        # legible against the illustrative anatomical image.
        for radius,colour in ((r+8,(36,120,150)),(r+4,(44,190,205)),(r,orange)):
            d.ellipse((x-radius,y-radius,x+radius,y+radius),outline=colour,width=2)
        d.ellipse((x-3,y-3,x+3,y+3),fill=(255,225,125));d.text((x-22,y+12),name,fill=(210,225,230))
    d.text((662,454),'Illustrative brain graphic; activity is real MaleCNS output',fill=(130,150,165))
    return canvas

def main():
    # Reproduce the longest completed fresh v5 run, which died at 10.34 s.
    import doom.fearless.doom_guy_train as trainer
    trainer.CAP=30
    policy=ActorCritic(2*len(groups()));saved=np.load(CHECKPOINT);policy.actor[:]=saved['actor'];policy.value[:]=saved['value'];frames=[]
    def hook(*args):
        if args[0]%3:frames.append(render(*args))
    result=run_episode(SEED,policy,'doom_guy',False,920000+SEED,deterministic_eval=False,frame_hook=hook,show_hud=True)
    OUT.parent.mkdir(parents=True,exist_ok=True);frames[0].save(OUT,save_all=True,append_images=frames[1:],duration=83,loop=0,optimize=False)
    print({'output':str(OUT),'frames':len(frames),'metrics':result['metrics']})

if __name__=='__main__':main()
