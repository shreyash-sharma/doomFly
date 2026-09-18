# DoomFly: how it works

DoomFly is a character: a fruit fly who thinks he's Doom Guy. This page covers the machinery underneath: the data, the simulation, the controller, the aggression neurons, the tests, and the parts that didn't work.

**Short version:** the wiring is real, and almost everything that turns wiring into gameplay is an engineering choice. Read it that way.

---

## 1. The connectome

DoomFly runs on **MaleCNS v1.0**, released on September 3, 2026 by HHMI Janelia with Google Research, MRC LMB and the University of Cambridge ([Cell paper](https://doi.org/10.1016/j.cell.2026.08.015), [Google Research post](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)). It's a reconstruction of a male *Drosophila* brain, optic lobes and ventral nerve cord from electron microscopy.

What we import:

| | |
|---|---|
| Retained neurons | **166,700** (every entry with an assigned superclass; explicit glia excluded) |
| Directed neuron-to-neuron edges | **25,582,938** |
| Synaptic contacts behind those edges | **124,177,617** (published confidence threshold 0.5) |
| Cropping / pruning | none: no weight threshold, self-connections kept |
| Runtime graph SHA-256 | `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b` |

The release reports ~166,000 neurons and ~125M synapses. Our "25.6M connections" counts neuron pairs, not synapses.

Source files are pinned by URL and SHA-256 in `doom/datasets.json` and `data-provenance/malecns_v1/source.lock.json`.

## 2. From pixels to neurons

Each ViZDoom frame is sampled into the fly's photoreceptor inputs:

- **3,335 R1–R6 brightness inputs**
- **811 R8 color inputs**

The pixel-to-receptor positions and color responses are **inferred proxies**, not a measured model of the fly eye.

## 3. The dynamics

Activity propagates across the full graph in a native C++ kernel (`doom/build_kernel`), using simplified spiking/rate dynamics with neurotransmitter-signed weights. The dynamics are **chosen**, not fitted to recordings. It's a model running on real wiring, not a living brain.

## 4. The controller

This is the part that turns neurons into Doom Guy.

**Inputs (neural only).** Firing rates from 20 populations:
- 14 descending-neuron and motor readouts (DNa02, DNp09, DNp20, DNpe017, MDN, MN9)
- LC4 and LPLC2, left and right (looming/escape-associated visual projection neurons)
- Tk-FruM (see §5)
- pC2l (a direct Tk-FruM target)

Each is summarised over an 18-step history as a mean and a trend, which gives 40 features.

**Readout.** A linear softmax policy over 11 action macros: noop, advance, retreat, left, right, attack, advance+attack, strafe-left+attack, strafe-right+attack, orbit-left+attack, orbit-right+attack. Actions are held for 4 ticks and sampled stochastically with fixed per-seed random streams.

**Training.** Clipped PPO on the readout only. The connectome and dynamics stay frozen. v5 trained for **12 episodes over 3 seeds**, with 5-second episodes. That's a very small amount of training.

**Reward.** Kills, hits and damage dealt count positively. Damage taken, death, time and ammo spent count negatively.

### ⚠️ What the controller can indirectly "see"

The readout never receives game coordinates or button suggestions. But two inputs come from **game state, not pixels**:

1. **Threat current.** When an enemy is inside the forward 120° field, current is injected into LC4/LPLC2 on the side matching its direction, scaled by proximity. The readout reads LC4/LPLC2 left and right. So the direction of the nearest enemy reaches the controller **through stimulated neurons, one step removed**.
2. **Tk-FruM current** (§5) is triggered by enemies, damage dealt, damage taken and kills, and Tk-FruM is also a readout input.

We added this bridge because the pure visual pathway failed (§7). It's the biggest reason the gameplay works, and the main reason **"the fly sees the demons" isn't an accurate description**.

## 5. The aggression neurons

**Identity.** MaleCNS annotations include an exact five-neuron match to the male-specific tachykinin/fruitless population, `AVLP727m`, annotated **"Asahina 2014: TK-FruM"** (body IDs 18792, 18987, 50960, 530190, 533662). Asahina et al. ([Cell 2014](https://doi.org/10.1016/j.cell.2013.12.045)) showed that activating these neurons increases intermale aggression in real flies.

**Doom Guy mode** injects current into those five neurons (amplitude ≤10) when an enemy is present or damage or a kill happens.

**Controls:**
- **No Tk:** same controller, no Tk-FruM current
- **Random control:** the same current schedule delivered to five predetermined unrelated neurons

**What happened before v5 (important):**
- Under the original fixed controller, stimulating Tk-FruM was **anti-aggressive**: less forward movement, fewer attacks, more damage taken. Constant stimulation cut attack ticks from ~165 to ~20.
- Tracing downstream found pC2l, which is associated with courtship-like pursuit, not strikes. A pC2l-to-forward adapter produced movement but **zero attacks**.
- No biologically supported "strike" output was found downstream of Tk-FruM.

So the aggression in DoomFly isn't the fly's circuit producing Doom combat by itself. v5's **trained readout** learned how to use Tk-FruM activity, along with everything else it reads.

## 6. Results (v5, frozen)

### Final test
Ten untouched seeds (787001–787010), run once, 5-second cap.

| Condition | Total kills | Mean damage dealt | Mean damage taken | Survival |
|---|---:|---:|---:|---:|
| Tk-FruM stimulated | 7 | 25.5 | 58.8 | 4.95 s |
| Random neurons stimulated | 6 | 24.0 | 55.8 | 5.00 s |
| No stimulation | 5 | 24.0 | 59.7 | 4.95 s |

Forward, turn and strafe activity were almost identical across conditions.

**Reading:** Tk-FruM had the most kills, but it beat random stimulation by **one kill** and was no better on survival or damage taken. This is **not evidence of a circuit-specific aggression effect.**

### 30-second runs (Tk-FruM, seeds 788001–788003)

| Seed | Kills | Hits | Damage dealt | Damage taken | Shots | Survival |
|---|---:|---:|---:|---:|---:|---:|
| 788001 | 2 | 4 | 50 | 114 | 194 | 10.3 s |
| 788002 | 1 | 0 | 0 | 102 | 116 | 6.4 s |
| 788003 | 2 | 0 | 0 | 102 | 128 | 7.4 s |

The controller stays active: it moves, turns, strafes and fires throughout. It dies within about 10 seconds. The recorded hits and damage are low relative to kills, so treat the hit counters with caution.

Checkpoint: `outputs/fearless/doom-guy-fly-v5-strafe/doom-guy-decoder.npz`
SHA-256 `5e02c81cfb9da9fbe703519f33bccd832003206c9a266b25777079bb88532dc7`

## 7. What failed

- **The looming gate.** Before any gameplay claims, we pre-registered a test: do expanding shapes recruit LC4, the fly's looming detectors? In our model, **LC4 fired zero spikes** in all 8 conditions. An external FlyVis optic-flow check also failed its pre-registered readout. That's why threat input is injected (§4), not seen.
- **Tk-FruM under the fixed controller** made the fly worse at fighting (§5).
- **The pursuit adapter** moved forward but never attacked.
- **v6** added hitscan-priority and wasted-ammo reward shaping. It **underperformed v5** and is kept as a negative result.

## 8. What DoomFly is and isn't

**It is:** a character-driven engineering demo that loads a complete released connectome, drives it with game frames, stimulates a genuinely identified aggression-linked population, and uses a small trained readout to play Doom.

**It isn't:**
- a living or conscious fly
- natural fly vision (threat direction is injected)
- evidence that the aggression circuit makes a better fighter
- a fly that learned Doom (only the readout is trained)

## 9. Reproduce it

This repository holds the data hashes, the kernel, the controller, per-seed JSON event logs and the failed experiments. The full simulation needs Python 3.11, a C++ compiler and several GB of RAM. The trained v5 checkpoint is not included in this release.

---

*DoomFly is a fan parody, not affiliated with or endorsed by id Software, Bethesda or ZeniMax. MaleCNS data © its authors under their release terms.*
