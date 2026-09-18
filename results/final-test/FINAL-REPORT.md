# DoomFly v5 — finalized

## Headline

We connected a frozen MaleCNS simulation to Doom through an engineered neural
interface and trained a small PPO readout to control movement, turning, genuine
lateral strafing, and firing. The resulting DoomFly actively runs, circles,
shoots, survives most short encounters, and kills enemies on unseen seeds.

## Frozen final test

The checkpoint was evaluated once on ten untouched seeds (787001–787010).
Actions were sampled from the frozen learned policy with fixed per-seed random
streams. Episodes lasted at most five seconds.

| Condition | Total kills | Mean kills | Damage inflicted | Damage received | Forward ticks | Turn ticks | Strafe ticks | Survival |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoomFly + Tk-FruM | 7 | 0.70 | 25.5 | 58.8 | 60.8 | 60.4 | 56.0 | 4.95 s |
| Normal DoomFly | 5 | 0.50 | 24.0 | 59.7 | 60.5 | 60.1 | 56.4 | 4.95 s |
| Random stimulation | 6 | 0.60 | 24.0 | 55.8 | 60.8 | 60.0 | 56.7 | 5.00 s |

The values other than total kills are means across ten seeds. The Tk-FruM label
produced the largest kill count, but its advantage over matched random
stimulation was only one kill and it did not improve survival or damage received.
This is not evidence for a circuit-specific aggression effect.

## Frozen implementation

- Checkpoint SHA-256:
  `5e02c81cfb9da9fbe703519f33bccd832003206c9a266b25777079bb88532dc7`
- MaleCNS dynamics remained frozen; PPO trained only the small action readout.
- Decoder inputs contained MaleCNS activity and temporal history only.
- Doom observer state generated labelled artificial LC4/LPLC2 and Tk-FruM
  stimulation, never direct button commands.
- The action set included genuine lateral strafe and orbit-and-fire macros.
- Four-tick action persistence produced smoother combat motion.
- V6 hitscan-priority/ammunition shaping was rejected after it underperformed v5.

## Honest interpretation

V5 is the finalized DoomFly demo: a connectome-based neural controller with a
learned embodiment layer that performs recognizable, aggressive Doom gameplay.
It is still a modest player, not a high-skill Doom bot. `Doom Guy`, `aggressive`,
and related names are experiment labels and do not imply anger, intention,
consciousness, or natural visual understanding.
