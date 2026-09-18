"""Reproducibly package DOOMFLY's combat environment; no neural policy here.

Requires the official ZDoom ACC compiler only when rebuilding the checked asset.
Runtime uses the generated WAD. The old stock arena and learning experiments
remain separately versioned; their performance is not comparable to this map.
"""
import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = Path(__file__).parent / 'scenarios'
RULES = {
    'revision': 'combat-survival-v2',
    'arena_size_units': [768, 768],
    'initial_enemies': 4,
    'maximum_alive_enemies': 8,
    'maximum_alive_per_type': {'DoomFlyImp': 4, 'Zombieman': 4},
    'spawn_interval_game_seconds': 3,
    'spawn_rule': 'One attempt every 105 engine tics; occupied pads are skipped. Alternate types when possible, filling the other type when one reaches its four-enemy cap. At most eight living enemies.',
    'enemy': 'DoomImp (30 health, native projectile/melee AI, Clip drop) and native Zombieman (20 health, hitscan attacks, Clip drop); at most four of each',
    'ammo': 'Four ClipBox pickup pads; each respawns eight game seconds after collection. Enemy kills drop a Clip.',
    'initial_bullets': 100,
    'episode_end': 'player death only; no time limit or level exit',
    'neural_state': 'preserved across arena resets; baseline synaptic weights are fixed',
    'observation_boundary': 'Enemy counts, item counters and other engine state are spectator telemetry only. Controller input remains RGB pixels.',
    'comparison': 'New environment revision; do not pool with combat-survival-v1, defend_the_center or blue-floor learning experiments. V1 permits eight projectile-only survivors and a circling stalemate.',
    'sources': ['https://github.com/ZDoom/acc', 'https://vizdoom.farama.org/'],
}


def map_text():
    text = ['namespace = "ZDoom";']
    for x, y in [(-384, -384), (-384, 384), (384, 384), (384, -384)]:
        text.append(f'vertex {{ x={x}.0; y={y}.0; }}')
    text.append('sector { heightfloor=0; heightceiling=192; texturefloor="FLOOR0_5"; textureceiling="CEIL4_1"; lightlevel=224; }')
    for i in range(4):
        text.append('sidedef { sector=0; texturemiddle="STARTAN3"; }')
        text.append(f'linedef {{ v1={i}; v2={(i+1)%4}; sidefront={i}; blocking=true; }}')
    text.append('thing { x=0.0; y=0.0; angle=0; type=1; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; single=true; }')
    return '\n'.join(text).encode()


def write_wad(path, entries):
    offset = 12
    body, directory = bytearray(), bytearray()
    for name, data in entries:
        directory.extend(struct.pack('<ii8s', offset, len(data), name.encode().ljust(8, b'\0')))
        body.extend(data)
        offset += len(data)
    path.write_bytes(struct.pack('<4sii', b'PWAD', len(entries), offset) + body + directory)


def build(acc):
    acc = Path(acc).resolve()
    source = SCENARIOS / 'combat_survival.acs'
    with tempfile.TemporaryDirectory(prefix='doomfly-acs-') as temp:
        compiled = Path(temp) / 'BEHAVIOR.o'
        subprocess.run([str(acc), '-i', str(ROOT / 'tools/acc'), str(source), str(compiled)], check=True)
        wad = SCENARIOS / 'combat_survival.wad'
        write_wad(wad, [('DECORATE', (SCENARIOS / 'combat_survival.decorate').read_bytes()),
                        ('MAP01', b''), ('TEXTMAP', map_text()), ('BEHAVIOR', compiled.read_bytes()),
                        ('SCRIPTS', source.read_bytes()), ('ENDMAP', b'')])
    files = [Path(__file__), source, SCENARIOS / 'combat_survival.decorate',
             SCENARIOS / 'combat_survival.cfg', wad]
    manifest = {**RULES, 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'compiler': {'source': 'https://github.com/ZDoom/acc',
                             'commit': subprocess.check_output(['git', '-C', str(ROOT / 'tools/acc'), 'rev-parse', 'HEAD'], text=True).strip()}}
    (SCENARIOS / 'combat_survival.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'wad': str(wad), 'sha256': manifest['sha256'][wad.name]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acc', default=str(ROOT / 'tools/acc/build/acc'))
    build(parser.parse_args().acc)
