# FEARLESS experiment report — exploratory continuation

## Outcome

The fixed-weight MaleCNS/DOOMFLY model did **not** pass the preregistered LC4
synthetic looming gate. LC4 emitted no spikes for any expanding-disc stimulus or
matched control. The original protocol required stopping before the smoke matrix.
The project owner subsequently authorized an explicit deviation to continue the
Doom experiment as exploratory work. The failed gate remains a binding
limitation: no behavioral result will be described as an effect of visual looming
or of visually recruited LC4.

This is a clean negative model-validation result: the present retinal/LIF model
does not recruit the named LC4 population under the tested looming stimuli or the
recorded seed-41027 Doom sequence. It is not evidence that biological LC4 neurons
fail to respond to looming.

## Evidence

- Data audit passed for 166,700 neurons, 25,582,938 directed edge rows, and
  124,177,617 contacts. Runtime graph SHA-256:
  `346b8af85a11af13b8324e18669812c1924569e7d1adcb4e6f45cc461a2c344b`.
- Exact targets: LC4 126, LPLC2 185, and two each of DNp09, DNp01/GF, DNp20,
  and DNpe017. LC4 has 126 direct DNp01 edge rows representing 6,362 contacts.
- The outgoing-synapse clamp changes exactly 25,477 LC4 edge rows. The frozen
  126-neuron matched control changes 27,783 rows. Nodes, CSR topology, sensory
  arrays, and all nonselected weights remain unchanged.
- Eight frozen speed/location combinations were crossed with seven stimulus
  types (56 intact fresh-brain runs). Expansion produced 35 distinct retinal
  vectors in an inspected sequence; its first-to-last maximum receptor change
  was 0.44669, confirming nonconstant model input.
- LC4 median baseline-subtracted rate was 0.0 Hz/neuron for expansion, reversed
  recession, final-size static, spatial scramble, mean-luminance, blank, and
  blank-without-lamina controls. Zero of eight expansions passed the required
  per-combination 1 Hz/neuron margin. The registered thresholds were 1
  Hz/neuron expansion recruitment, a 1 Hz/neuron margin over every control, and
  success in at least 70% of combinations.
- Across expansion runs, LC4, LPLC2, DNp09, and DNp01 emitted zero spikes.
  DNp20 emitted 621 and DNpe017 emitted 462 total spikes, so the simulation and
  population accounting were active.
- A recorded 10-second, 350-tick seed-41027 intact Doom run supplied the exact
  same retinal vectors offline to fresh intact, LC4-silenced, and matched-control
  brains. Input hashes, source-frame hashes, and timing matched on every tick.
  All three event files were byte-identical. LC4, LPLC2, DNp09, and DNp01 again
  emitted zero spikes; DNp20/DNpe017 emitted 685/519 spikes in every condition.
  Consequently, every paired named-population contrast was zero.
- Two fresh one-second seed-41027 executions of the final runner produced 36
  identical tick-zero/tick records after excluding run IDs and wall timing.
  Comparison SHA-256:
  `6659dbf046bc65d55c8ecde68b285207ce781b1edc65d6fa06aca0acc2298a31`.
- Final combined test run: 108 passed, two platform skips, and 63 dependency
  warnings; no failures.

## Protocol disposition

Completed: source/data verification, baseline reproduction, target resolution,
matched-control freezing, intervention validation, bounded runner, raw-event
metrics, determinism gate, synthetic looming gate, and recorded-Doom same-input
diagnostic.

After an explicit owner-authorized deviation, the public VIDraft/fruitfly-brain
artifact was audited first. The direct-drive bridge, 3 × 3 development smoke,
and 3 × 10 held-out pilot then completed. This later work is explicitly
exploratory and tests the fixed outgoing-weight intervention under artificial
current—not LC4 looming mechanism or natural LC4 recruitment.

The follow-up visual-pathway audit found that every LC4 cell is structurally
reachable from the mapped retina in three hops, but only five of 20,664 direct
LC4 presynaptic neurons fired during the representative looming replay. A
parameter-free graded OFF-contrast proxy beat every nonblank control in only
1/8 frozen speed/location combinations. It therefore failed the original 70%
selectivity criterion. No new natural-vision Doom cohort was opened; that phase
requires independently fitted graded motion/contrast dynamics rather than a
gain increase or hand-authored looming detector.

An external, isolated audit of three official pretrained FlyVis optic-flow
checkpoints used the exact same 56 frozen synthetic frames (all frame hashes
matched). Its preregistered broad T4/T5 motion-energy readout failed 0/8
controls for each checkpoint. A narrower exploratory spatial-flow readout did
find outward local flow for expanding discs in 8/8, 6/8, and 3/8 combinations
across checkpoints `000`, `001`, and `002`, respectively. That instability,
and FlyVis's lack of LC4 cells and incompatible female-connectome geometry,
means it does not license a current bridge into MaleCNS. No FlyVis-derived
signal was sent to MaleCNS or Doom.

## VIDraft audit and direct-drive bridge

The public VIDraft static artifact was audited before resuming any Doom work.
Its baked data reproduce the advertised LC4 arithmetic exactly: 0.839661 intact
to 0.091226 after LC4 lesion (−89.135%). It directly drives LPLC2+LC4 and ships
pre-measured outputs; weights and engine source are not public. LC11 and LC6
control activity snapshots are byte-identical to intact. The final LC4-lesion
snapshot retains three nonzero LC4-labeled activity entries, so the public files
do not independently determine whether its lesion is full cellular silence or an
outgoing-output clamp.

To bridge models without changing weights or dynamics, Doomfly received a fixed,
separate direct-current sweep into LC4+LPLC2. At current 20, intact DNp01 had
536 drive-window spikes; LC4 output clamp had 461 and the matched control 524.
This shows that direct drive can engage the Doomfly downstream pathway even
though its retinal input does not recruit LC4. One disclosed development Doom
seed then used the same nonvisual current: intact/silenced/matched restricted
survival was 9.114/8.571/5.600 seconds. This single closed-loop observation is
diagnostic only; it is not a behavioral effect estimate and cannot be attributed
to looming.

## Reproducibility artifacts

- `target-manifest.json`: exact identities, sides, connectivity, control match.
- `intervention-manifest.json`: selected indices, affected rows, weight hashes.
- `synthetic/report.json` and `synthetic/retinal-input-suite.npz`: complete
  continuous gate results, frame/input hashes, parameters, and exact vectors.
- `open-loop/source/`: bounded closed-loop source run and retinal archive.
- `open-loop/report.json` plus per-condition JSONL: offline paired replay.
- `vision-audit/report.json` and `vision-audit/graded-proxy-controls.png`:
  structural reachability, functional bottleneck, and graded-proxy controls.
- `flyvis-audit/` and `flyvis-radial-audit/`: isolated official pretrained
  optic-flow checks, exact-frame verification, and their failed/stability-limited
  output gates; neither directory contains a MaleCNS or Doom run.
- `WORKLOG.md`: setup commands, benchmark, and staged decisions.

## Caveats

The visual front end is an explicitly coarse luminance/LIF proxy, not a calibrated
fly retina or physiological LC4 model. The negative result localizes the failure
to recruitment in this model configuration; it does not establish whether the
cause is retinal projection, sign/dynamics, missing visual preprocessing,
cell-type mapping, or intrinsic/synaptic parameters. Those are follow-up model
development questions and were not tuned after observing the gate.

## Final exploratory direct-drive outcome

The owner-authorized bridge and held-out pilot are now complete. Under constant
direct LC4+LPLC2 current 20, LC4 outgoing silencing increased paired 30-second
restricted survival by a mean 12.35 seconds (10,000-resample 95% CI 7.63–16.87;
n=10), versus +0.51 seconds (−1.08–2.24) for the matched control. Median survival
was 7.77 seconds intact, 19.64 seconds with the LC4 clamp, and 8.90 seconds with
the matched clamp. All 30 runs validated, two LC4-clamp runs were capped, and no
run failed or was excluded.

The same-input neural comparison anchors the mechanism inside the model: LC4
clamping reduced DNp01 from 4,861 to 3,662 spikes (−24.7%), while the matched
control produced 4,858 (−0.06%). The chosen BCI consequently moved forward and
attacked less and turned more. This supports an LC4-dependent, directly driven
model circuit and a large effect on this engineered Doom controller. It does not
rescue the failed natural retinal/looming gate and is not evidence about fear.

Full tables, bootstrap results, figures, captions, and limitations are in
`outputs/fearless/direct-doom/pilot/REPORT.md`.

## Event-triggered artificial-threat follow-up

After the natural-vision path was explicitly closed, a separate artificial
observer-state channel was tested. It maps documented Doom threat events to
LC4/LPLC2 current only; it cannot write actions, and all 60 held-out runs had
identical requested and applied BCI buttons. The one-seed validation gate passed
and the 10-seed × 6-condition cohort completed without exclusions.

The channel drove LC4/LPLC2 and direction-matched turns but did not improve
survival over uninformed Doomfly: mean paired threat-informed minus uninformed
restricted survival was −0.46 s (95% bootstrap CI −1.02 to −0.09; 10-s cap).
Threat-informed did outperform its LC4-output clamp (+3.27 s mean) and the
false-alarm stream (+1.33 s mean), while the neural activity-rate-matched
control also performed poorly. Thus timing and LC4 output influence use of the
artificial current, but the resulting behavior was not effective defensive
handling. See `outputs/fearless/event-threat/pilot/REPORT.md`.

## DOOM GUY FLY pre-cohort stop

An exact five-neuron MaleCNS Tk-FruM/AVLP727m mapping was independently
identified from released annotations and linked to primary aggression-arousal
literature. Two fresh-seed smoke tests confirmed direct stimulation changed both
the target and BCI downstream neurons, but consistently *reduced* forward
movement and attack while increasing damage. Rather than tune the stimulation
to chase an aggressive phenotype, the proposed cohort was stopped before launch.
See `outputs/fearless/aggression/REPORT.md`.

The later downstream audit found a Tk-FruM-responsive direct pC2l pursuit
candidate, but no supported lunge/strike pathway. A pC2l-forward-only engineered
adapter passed its neural-specificity control but had zero attacks and high
damage in two smoke seeds, so no full adapter cohort was launched.
