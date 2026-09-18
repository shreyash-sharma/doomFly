# Fixed-weight baseline reference

This document describes the original baseline, including its lack of plasticity.
The current live experiment uses `--model experimental-v6 --learning`; see the
[current protocol](../docs/doom-live-training.md) and [repository setup](../README.md).
Those later changes supersede the baseline-only statements below.

# Fly Brain / Doom

This is a live whole-connectome **neural BCI experiment**, not a validated fly
emulation or a trained Doom player. The fixed visual-neuron BCI turns, moves and
fires in ViZDoom. A comprehensive review found and corrected a refractory-event
error; old kill counts are legacy results, not validation of the current model.
See [the neuroscience review](../docs/doom-neuroscience-review.md) and
`outputs/doom/audit/experiments.json` for matched controls and scientific limits.
These checks do not establish skill, learning, or natural motor semantics. The biological-role
comparison (DNa02, DNp09, MDN, MN9) remains silent in the initial visual test.

## What runs

- One complete retained MaleCNS v1.0 graph: 166,700 neurons, 25,582,938 directed
  edge rows, 124,177,617 contacts. All original retained weak and self edges remain.
- Live ViZDoom 1.3.0, using its packaged freely distributed game/scenario assets.
  It is a Doom-engine scenario, not a copy of the commercial DOOM campaign.
- The public broadcaster now defaults to the original `combat_survival` map:
  four enemies (alternating DoomImp with 30 health and native Zombieman with 20), one additional spawn attempt every
  three game seconds, capped at eight living enemies. Four 50-bullet ClipBox
  pickups return eight seconds after collection. Enemies drop a Clip on death.
  Each round starts with 100 bullets and ends only at player death, with no time
  limit. Neural state continues between rounds; synaptic weights remain fixed.
  Enemy spawning, pickups and death resets are engineered environment rules,
  not neural behaviors. Enemy counts never enter the neural decoder.
  The old `defend_the_center` scenario remains available for reproduction with
  `--scenario defend_the_center`; historical audits and learning experiments
  must not be pooled with this different arena.
- RGB screen samples enter 3,335 of 3,377 R1–R6 photoreceptors. Their columns are
  inferred from all released contacts onto hex-annotated L1/L2/L3. 42 unmapped
  receptors remain in the graph without externally assigned pixels. Color and
  absolute viewing angles are not calibrated. The screen is an experimental
  overlapping left/right viewport, not a measured eye model.
- Photoreceptor current uses a 10 ms low-pass luminance, 30 mV maximum current
  equivalent and I/(0.02+I) saturation, with sRGB linearization. L1/L2/L3/L5
  receive a declared 12 mV tonic current equivalent. These are **chosen model
  parameters**, not measured/calibrated male-fly physiology. They are needed to
  explore inhibitory photoreceptor inputs in this coarse spiking model.
- All neurons use an approximate LIF model, including visual cells that really
  use graded signaling. tau_m=20 ms, tau_g=5 ms, rest=-52 mV, threshold=-45 mV,
  refractory=2.2 ms, delay=1.8 ms, contact gain=0.275 mV, dt=0.1 ms.
- ACh +; GABA/glutamate/histamine −; uncertain/modulator-only/conflicting signs
  use +. There are 3,718 uncertain-sign cells. This is not receptor physiology.
- The default BCI decoder receives only spike counts: DNp20 R−L turns with
  gain 0.12 degrees/tic/Hz (clamped ±6); summed DNpe017 rate moves with gain
  0.4 (clamped 0–20); a DNpe017 spike presses attack for one game tic. Neurons
  were selected after visual-response calibration, not in a preregistered test.
  These are chosen engineering mappings, not claims about biological motor
  function. `--decoder biological` keeps the DNa02/DNp09/MDN/MN9 comparison.
  Both sets of readouts remain visible. No scene recognition,
  navigation, enemy coordinates, additional aiming policy, or replacement policy
  enters the decoder. Native Doom weapon rules remain in the game engine.
- Optional positive-score-contingent 200 ms LB3c sugar input is implemented but
  OFF by default and in the public broadcast. An exploratory reward-enabled run
  showed strong perturbation and later quiet BCI readouts; it does not establish
  learning or isolate reward as the sole cause. No synaptic plasticity exists: this is stimulus
  delivery, **not training**, and no learning result is claimed.
- Neural time and Doom time stay aligned to <0.1 ms, even when computation is
  slower than wall time. No neural steps are skipped or relabeled as real time.
  The viewer shows the measured speed. All spectators see one shared run.

## Reproduce

Use the existing isolated Python 3.11 environment and full imported graph.

```sh
.venv-neural/bin/python -m pip install -r doom/requirements.txt
.venv-neural/bin/python -m doom.prepare
.venv-neural/bin/python -m doom.audit_data
.venv-neural/bin/python -m doom.build_kernel
.venv-neural/bin/python -m pytest tests/test_doom.py tests/test_doom_reference.py tests/test_connectome.py -q
.venv-neural/bin/python -m doom.audit_experiments
.venv-neural/bin/python -m doom.server
```

The build helper selects the platform suffix, replaces the library atomically,
and writes source/binary hashes. The runtime rejects a stale or modified binary.
On Linux the helper builds `libneural.so`.
Do not run many full-graph processes concurrently on a small machine.
`--condition` permits named ablations; they are labeled in every frame. The
intact graph is never silently pruned. `--reward sugar` enables stimulus delivery.
The local service binds **only 127.0.0.1:8766**, with GET /state and GET /health.
There is no remote control, filesystem serving, secret exposure or write API.
A broadcaster audit log records source-frame, input and spike hashes plus all
requested/applied actions for every game tic. Audit storage rotates at 20 MB with
four backups (100 MB total); run IDs distinguish restarts.
The new runtime also retains every step in hourly gzip archives, so a multiday
run does not lose its early evidence when the recent log rotates. Check disk
capacity and back up these archives before promising continuous operation.
`--checkpoint-dir PATH --resume --checkpoint-seconds 300` enables atomic neural
checkpoints. Recovery retains the saved neural/decoder state but starts a fresh
arena and new linked run ID; the interrupted round is censored. This format is
for the fixed baseline only, not the candidate learning classes. See
`deploy/doomfly/README.md` for the tested boundaries and prepared cloud setup.

The spectator transport captures up to eight real frames per wall second and
publishes one-second immutable batches through `/index` and
`/segments/RUN/SECOND`. The CDN shares those batches; the browser plays their
original capture timestamps with about 2.5 seconds of delay. Frame, action and
neural readouts remain coupled. `/state` stays available for audit tools and
legacy clients. Serialization happens once per capture, independent of viewer
count. No neural steps are skipped or replaced by display interpolation.
New combat logs also record the scenario and post-action game counters. Use
`--audit-dir` to isolate independent runs. The generated WAD, ACS source,
DECORATE actor and compiler/source hashes are in `doom/scenarios/`. Rebuild with
the official ACC compiler via `python -m doom.combat_arena --acc PATH_TO_ACC`;
its include files are expected in `tools/acc`. No compiler is needed at runtime.
`tests/test_doom_combat.py` checks the actual engine: death/reset, continuing
spawns, no 60-second cap, finite enemy population, collectible and replenishing
ammo, shooting and drops. Test-only console commands isolate environment rules;
they are never called by the broadcaster or presented as neural behavior.

`doom-ui` is the public spectator viewer.
Set its hosted `DOOM_STREAM_ORIGIN` to the read-only broadcaster HTTPS origin.
A temporary Cloudflare tunnel is suitable for this live development experiment,
not an always-on production service. The broadcaster and tunnel must stay alive;
when the host sleeps/disconnects the site explicitly goes offline. Production
24/7 operation requires an always-on compute host and stable tunnel/domain.

## Evidence and limits

- Graph and model framework: https://www.nature.com/articles/s41586-024-07763-9
- Male visual columns: https://www.nature.com/articles/s41586-025-08746-0
- Graded visual modeling: https://www.nature.com/articles/s41586-024-07939-3
- Histamine/receptor physiology: https://www.nature.com/articles/s41586-023-06681-6
- DNa02 steering: https://elifesciences.org/articles/102230
- Descending control: https://www.nature.com/articles/s41586-024-07523-9
- MN9 mouthpart control: https://elifesciences.org/articles/19892
- Visual reinforcement: https://elifesciences.org/articles/02395

The efficient kernel skips only subthreshold evolution that cannot produce a
spike without an input event, then analytically materializes it at the next event.
No weak edges or inactive neurons are removed from the graph. Toy-network tests
compare counts/voltages to a dense reference with changing inputs and inhibition.
Full-graph interventions test retinal causality and total disconnection, including
matched BCI readouts. A 210-tic live closed-loop test recorded actual game actions. These
are numerical/causal implementation checks, not biological validation.
