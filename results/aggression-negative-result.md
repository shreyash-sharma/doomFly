# DOOM GUY FLY — Tk-FruM pre-cohort stop report

## Biological identity

The MaleCNS annotations contain an exact five-neuron mapping to the
male-specific tachykinin/fruitless population reported by Asahina et al.:
`AVLP727m`, with the annotation synonym **“Asahina 2014: TK-FruM”**. The retained
body IDs are 18792, 18987, 50960, 530190, and 533662; graph indices are 8017,
8196, 36409, 134363, and 136053. All are acetylcholine-labelled,
`fru_high`, male-specific central-brain neurons; one is preliminarily traced.

Asahina et al. found that activating male-specific FruM+ tachykinin neurons
increased intermale aggression and that silencing them reduced it (Cell 2014,
DOI: 10.1016/j.cell.2013.12.045). P1 has independent primary evidence for a
persistent state enhancing aggression (Hoopfer et al., eLife 2015,
DOI: 10.7554/eLife.11346), but no P1 subset was selected because the present
annotations did not provide a comparably narrow direct mapping. The exact
annotation audit is [`audit.json`](audit.json).

This establishes a biological association for the label only. It does not show
that Doom enemies are fly rivals or that current injection recreates a peptide
or social circuit mechanism.

## Preregistered small smoke

Two fresh seeds (`513002`, `513003`) crossed normal, constant Tk-FruM current
10, and event-triggered Tk-FruM current (nearby/front enemy and recent damage;
amplitude at most 10). No condition wrote a game button: every action came from
the unchanged fixed BCI decoder. The full per-tick records and checks are in
[`smoke/report.json`](smoke/report.json).

| Seed | Condition | Tk spikes | DNp20 spikes | DNpe017 spikes | Attack ticks | Mean forward | Damage |
|---:|---|---:|---:|---:|---:|---:|---:|
| 513002 | Normal | 0 | 343 | 263 | 167 | 19.08 | 15 |
| 513002 | Constant Tk-FruM | 392 | 105 | 31 | 25 | 2.48 | 117 |
| 513002 | Event Tk-FruM | 118 | 330 | 247 | 157 | 18.25 | 15 |
| 513003 | Normal | 0 | 343 | 259 | 161 | 18.95 | 6 |
| 513003 | Constant Tk-FruM | 387 | 97 | 24 | 18 | 1.92 | 69 |
| 513003 | Event Tk-FruM | 245 | 194 | 136 | 82 | 9.90 | 30 |

## Stop decision

The required identity and stimulation checks pass: Tk-FruM spikes, downstream
decoder populations change, and decoded behavior changes. But the effect is
consistently **anti-aggressive under this fixed Doomfly BCI**: less forward
movement and attack, lower downstream decoder activity, and more damage. It is
therefore not valid to describe the resulting condition as creating aggressive
Doom behavior.

The requested 10-seed six-condition cohort was **not launched**. Increasing,
retuning, or selectively searching stimulation parameters after this result
would turn the experiment into outcome-directed tuning. A later study would
need a separately justified dynamical/neuromodulatory model and a frozen
behavioral validation criterion before using the “Doom Guy” label.

No claim is made about anger, rage, intention, consciousness, natural visual
perception, or biological fly behavior.

## Downstream pathway and aggression-action-adapter follow-up

The requested downstream trace identified two direct Tk-FruM targets,
`AVLP716m_L/R` (body IDs 12922 and 13723; graph indices 2734 and 3472). Their
annotations identify them as pC2l. Kohatsu et al. (Nature Communications 2015,
DOI: 10.1038/ncomms7457) supports pC2l as a priming node for courtship-like
following pursuit, not as a lunge/strike or Tk-FruM-specific aggression output.
The original Tk-FruM trace also reaches pCd- and pC1-labelled cells, but the
available primary evidence supports persistent state/social behavior, not a
direct aggression-to-action mapping. No supported lunge/strike population was
found downstream of the five Tk-FruM cells.

In a frozen 35-tick open-loop test, pulsed Tk-FruM generated 20 pC2l spikes,
versus 0 at baseline and 0 after a deterministic matched unrelated ACh
cb-intrinsic five-neuron drive. An equal-total temporally shuffled Tk-FruM
schedule generated 11 spikes. This validates pC2l as a narrow,
pursuit-associated *neural* readout, not an aggression readout. Details are in
[`pathway.json`](pathway.json).

I therefore constructed the only defensible adapter: pC2l spikes map to forward
movement using the original BCI's fixed forward gain; there is deliberately no
enemy input, turning rule, or attack rule. The separate scripted motor condition
is the only condition that uses enemy geometry and is labelled non-biological.
In two new smoke seeds, Tk-FruM drove more pC2l activity than matched random
(14/13 versus 3/4 spikes), and hence more adapter forward output (0.86/0.78
versus 0.24/0.34), but both adapter conditions had **zero attack ticks**. They
took substantial damage, unlike the original BCI. The smoke is saved at
[`adapter-smoke/report.json`](adapter-smoke/report.json).

The full cohort was not launched: the prerequisite neural readout existed, but
there is no biologically supported strike readout and the adapter does not make
an aggression index stronger than baseline in the required attack component.
Adding a firing mapping would be outcome-directed invention, not a validated
embodiment mapping.
