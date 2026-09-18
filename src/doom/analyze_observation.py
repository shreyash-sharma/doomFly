"""Offline descriptive QA; never changes or loads the running neural model."""
import argparse, collections, csv, gzip, json, math, os, statistics
from pathlib import Path

def records(path):
    if not path.exists():
        return
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def stats(values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    if not values:
        return None
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'max': max(values)}

def nonfinite(value):
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(nonfinite(v) for v in value.values())
    if isinstance(value, list):
        return any(nonfinite(v) for v in value)
    return False

def clipped(x, low, high):
    return max(low, min(high, x))

def expected_action(readouts):
    # Independently reconstruct the published BCI equations from recorded rates.
    left = sum(r['rate_hz'] for r in readouts if r['type'] == 'DNp20' and r['side'] == 'L')
    right = sum(r['rate_hz'] for r in readouts if r['type'] == 'DNp20' and r['side'] == 'R')
    move = sum(r['rate_hz'] for r in readouts if r['type'] == 'DNpe017')
    return {'turn': clipped((right - left) * .12, -6, 6),
            'forward': clipped(move * .4, 0, 20),
            'attack': any(r['spikes'] > 0 for r in readouts if r['type'] == 'DNpe017')}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if not (out / 'complete.json').exists():
        raise SystemExit('Observation is incomplete. No final analysis was produced.')
    protocol = json.loads((out / 'protocol.json').read_text())
    initial = json.loads((out / 'initial-state.json').read_text())
    completion = json.loads((out / 'complete.json').read_text())
    violations = collections.Counter()
    examples = collections.defaultdict(list)
    def check(name, valid, location):
        if not valid:
            violations[name] += 1
            if len(examples[name]) < 5:
                examples[name].append(location)

    rows, seen_episodes, run_ids, source_hashes = [], {}, set(), set()
    previous = None
    for record in records(out / 'snapshots.jsonl.gz'):
        s = record['state']; where = s['sequence']; c = s['clocks']; g = s['game']
        run_ids.add(s['run_id'])
        source_hashes.add(json.dumps(s['provenance'], sort_keys=True))
        check('nonfinite_snapshot', not nonfinite(s), where)
        check('jpeg_digest', record['jpeg_digest_valid'], where)
        check('clock_alignment', abs(c['neural_seconds'] - c['game_seconds']) <= .0002, where)
        check('frame_audit_alignment', s['input_frame_sha256'] == s['audit']['source_frame_sha256'], where)
        check('population_sum', sum(p['spikes'] for p in s['populations']) == s['window_spikes'], where)
        check('population_coverage', sum(p['neurons'] for p in s['populations']) == 166700, where)
        check('no_learning_mode', s['reward'] == {'mode': 'off', 'sugar_pulses': 0, 'active': False, 'plasticity': False}, where)
        check('configuration', s['condition'] == 'intact' and s['decoder'] == 'bci', where)
        check('retinal_bounds', all(0 <= x <= 1 for x in s['retina']['luminance']) and
              all(0 <= x <= 30.0001 for x in s['retina']['drive_mv']), where)
        check('retinal_display_alignment', len({len(s['retina'][k]) for k in
              ['uv', 'luminance', 'neuron_ids', 'side', 'filtered_luminance', 'drive_mv', 'spikes_last_step']}) == 1, where)
        for b in s['raster']['bins']:
            check('raster_spike_bound', b['window_ms'] > 0 and
                  all(0 <= x <= math.ceil(b['window_ms'] / 2.2) + 1 for x in b['counts']), where)
            check('raster_count_shape', len(b['counts']) == len(s['raster']['neuron_ids']), where)
            check('raster_population_subset', sum(b['counts']) <= b['population_spikes'], where)
        for p in s['populations']:
            expected_rate = p['spikes'] / p['neurons'] / (s['window_ms'] / 1000)
            check('population_rate', abs(p['mean_rate_hz'] - expected_rate) < .01, where)
        if previous and s['run_id'] == previous['run_id']:
            check('clock_monotonic', c['neural_seconds'] >= previous['clocks']['neural_seconds'], where)
            check('spike_counter_monotonic', s['total_spikes'] >= previous['total_spikes'], where)
            check('sequence_monotonic', s['sequence'] >= previous['sequence'], where)
        for e in [*s['episodes'], g]:
            key = (s['run_id'], e['episode'])
            prior = seen_episodes.get(key)
            if prior is None or e['tick'] >= prior['tick']:
                seen_episodes[key] = {**e, 'run_id': s['run_id'],
                    'last_seen_wall_seconds': record['elapsed_wall_seconds']}
        rows.append({'wall_seconds': record['elapsed_wall_seconds'], 'sequence': where,
            'neural_seconds': c['neural_seconds'], 'game_seconds': c['game_seconds'],
            'age_seconds': (record['received_epoch'] * 1000 - s['generated_at_ms']) / 1000,
            'speed_lifetime': c['speed'], 'brain_step_ms': c['brain_step_ms'],
            'episode': g['episode'], 'health': g['health'], 'kills': g['kills'], 'ammo': g['ammo'],
            'turn': s['action']['turn'], 'forward': s['action']['forward'],
            'attack': int(s['action']['attack']), 'total_spikes': s['total_spikes'],
            'population_hz': s['window_spikes'] / 166700 / (s['window_ms'] / 1000),
            'mean_luminance': statistics.mean(s['retina']['luminance']),
            'source_frame': s['input_frame_sha256']})
        previous = s
    if not rows:
        raise SystemExit('No valid snapshots. Observation cannot be evaluated.')

    events, prev_event = [], None
    max_decoder_error = {'turn': 0., 'forward': 0.}
    max_filter_error = 0.
    tick_gaps = []
    for record in records(out / 'events.jsonl.gz'):
        e = record['event']; where = e['tick']
        check('event_nonfinite', not nonfinite(e), where)
        check('event_configuration', e['run_id'] == protocol['run_id'] and e['condition'] == 'intact'
              and e['decoder'] == 'bci' and e['reward_mode'] == 'off', where)
        check('tic_interval', e['output_game_tick'] - e['input_game_tick'] == 1, where)
        check('neural_clock_schedule', abs(e['neural_ms'] - round(e['tick'] * 10000 / 35) * .1) < .0011, where)
        check('requested_applied', e['requested'] == e['applied'], where)
        check('reward_disabled', not e['sugar_applied'] and not e['sugar_scheduled_for_next_step'], where)
        expected = expected_action(e['readouts'])
        for k in ['turn', 'forward']:
            error = abs(expected[k] - e['applied'][k])
            max_decoder_error[k] = max(max_decoder_error[k], error)
            # Published rates are rounded to 0.001 Hz; the decoder uses full precision.
            check('decoder_' + k, error <= .001, where)
        check('decoder_attack', expected['attack'] == e['applied']['attack'], where)
        check('action_bounds', -6 <= e['applied']['turn'] <= 6 and 0 <= e['applied']['forward'] <= 20, where)
        if prev_event and prev_event['run_id'] == e['run_id']:
            if e['tick'] != prev_event['tick'] + 1:
                tick_gaps.append([prev_event['tick'], e['tick']])
            else:
                seconds = e['neural_interval_ms'] / 1000
                decay = math.exp(-seconds / .1)
                prev_rates = {r['id']: r['rate_hz'] for r in prev_event['readouts']}
                for r in e['readouts']:
                    predicted = prev_rates[r['id']] * decay + r['spikes'] / seconds * (1 - decay)
                    error = abs(predicted - r['rate_hz'])
                    max_filter_error = max(max_filter_error, error)
                    check('spike_rate_filter', error <= .0011, where)
        events.append({'tick': e['tick'], 'neural_seconds': e['neural_ms'] / 1000,
            'episode': e['episode'], 'attack': int(e['applied']['attack']),
            'turn': e['applied']['turn'], 'forward': e['applied']['forward'],
            'readout_spikes': sum(r['spikes'] for r in e['readouts']), 'reward': e['reward'],
            'source_frame': e['source_frame_sha256'], 'input_hash': e['input_sha256']})
        prev_event = e

    episodes = []
    for e in seen_episodes.values():
        if e['run_id'] != protocol['run_id'] or e['episode'] < initial['game']['episode']:
            continue
        e['left_censored'] = e['episode'] == initial['game']['episode']
        e['right_censored'] = not e['finished']
        e['duration_neural_seconds'] = e['tick'] / 35
        e['complete_in_window'] = not e['left_censored'] and not e['right_censored']
        episodes.append(e)
    episodes.sort(key=lambda e: e['episode'])
    full = [e for e in episodes if e['complete_in_window']]
    wall = rows[-1]['wall_seconds'] - rows[0]['wall_seconds']
    neural = rows[-1]['neural_seconds'] - rows[0]['neural_seconds']
    gaps = [b['wall_seconds'] - a['wall_seconds'] for a, b in zip(rows, rows[1:])]
    spike_delta = rows[-1]['total_spikes'] - rows[0]['total_spikes']
    thirds = []
    for i in range(3):
        low = rows[0]['neural_seconds'] + neural * i / 3
        high = rows[0]['neural_seconds'] + neural * (i + 1) / 3
        selected = [e for e in events if low <= e['neural_seconds'] < high]
        thirds.append({'third': i + 1, 'event_count': len(selected),
            'attack_fraction': statistics.mean(e['attack'] for e in selected) if selected else None,
            'turn': stats(e['turn'] for e in selected), 'forward': stats(e['forward'] for e in selected),
            'game_reward_sum': sum(e['reward'] for e in selected),
            'interpretation': 'Descriptive only: no learning intervention, reward sum is not kill count.'})
    start_hash = json.loads((out / 'fingerprints-start.json').read_text())
    end_hash = json.loads((out / 'fingerprints-end.json').read_text())
    source_match = all(start_hash[k] == v for k, v in initial['provenance']['source_sha256'].items())
    graph_match = start_hash['outputs/doom/malecns_v1/graph.npz'] == initial['provenance']['graph_sha256']
    public = list(records(out / 'public-curl-checks.jsonl'))
    errors = list(records(out / 'errors.jsonl'))
    known_kills = sum(e['kills'] for e in episodes) - initial['game']['kills']
    result = {
        'status': 'analyzer_preflight' if completion.get('not_a_complete_hour') else 'complete_descriptive_analysis',
        'protocol': protocol, 'collector': completion,
        'coverage': {'snapshots': len(rows), 'audit_events': len(events), 'run_ids': sorted(run_ids),
            'snapshot_wall_seconds': wall, 'neural_seconds': neural, 'window_speed': neural / wall,
            'max_sample_gap_seconds': max(gaps, default=0), 'tick_gaps': tick_gaps,
            'saved_frames': len(list((out / 'frames').glob('*.jpg'))),
            'first_event_tick': events[0]['tick'] if events else None,
            'last_event_tick': events[-1]['tick'] if events else None},
        'checks': {'violations': dict(violations), 'examples': dict(examples),
            'decoder_max_absolute_error': max_decoder_error, 'filter_max_absolute_error_hz': max_filter_error,
            'source_matches_running_provenance': source_match, 'graph_matches_running_provenance': graph_match,
            'disk_fingerprints_unchanged': start_hash == end_hash, 'runtime_provenance_variants': len(source_hashes),
            'full_live_weight_array_recorded': False, 'full_live_voltage_state_recorded': False},
        'game': {'episode_count_including_censored': len(episodes), 'complete_episodes': len(full),
            'complete_episode_kills': stats(e['kills'] for e in full),
            'complete_episode_duration_seconds': stats(e['duration_neural_seconds'] for e in full),
            'known_window_kill_increment': known_kills,
            'snapshot_empty_ammo_fraction': statistics.mean(r['ammo'] == 0 for r in rows),
            'snapshot_attack_with_empty_ammo_fraction': statistics.mean(r['ammo'] == 0 and r['attack'] for r in rows),
            'episodes': episodes},
        'neural': {'total_spike_increment': spike_delta, 'mean_spikes_per_neuron_hz': spike_delta / neural / 166700 if neural else None,
            'snapshot_population_rate': stats(r['population_hz'] for r in rows),
            'snapshot_luminance': stats(r['mean_luminance'] for r in rows),
            'distinct_sampled_input_frames': len({r['source_frame'] for r in rows}),
            'event_attack_fraction': statistics.mean(e['attack'] for e in events) if events else None,
            'event_turn_saturation_fraction': statistics.mean(abs(e['turn']) >= 5.999 for e in events) if events else None,
            'event_move_saturation_fraction': statistics.mean(e['forward'] >= 19.999 for e in events) if events else None,
            'neural_time_thirds': thirds},
        'public_feed': {'curl_samples': len(public), 'curl_successes': sum(p.get('ok', False) for p in public),
            'curl_failures': [p for p in public if not p.get('ok')],
            'curl_latency_seconds': stats(p['request_seconds'] for p in public),
            'curl_run_ids': sorted({p['state']['run_id'] for p in public if p.get('ok')}),
            'curl_frame_age_at_response_seconds': stats(
                p['received_epoch'] + p['request_seconds'] - p['state']['generated_at_ms'] / 1000
                for p in public if p.get('ok')),
            'local_snapshot_age_seconds': stats(r['age_seconds'] for r in rows),
            'primary_collector_errors': errors,
            'interpretation': 'Minute samples cannot prove continuous uptime. urllib 403 is a client transport result.'},
        'learning': {'plasticity_implemented': False, 'learning_established': False,
            'expected_progress': 'No training-driven improvement is predicted. Activity and episode variation are not learning.'},
        'scientific_verdict': 'These checks evaluate an approximate connectome-driven BCI; they cannot establish a literal fly brain, correct fly vision, understanding or learning.'}
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, data in [('timeseries.csv', rows), ('episodes.csv', episodes)]:
        if data:
            with (out / name).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
    try:
        os.environ.setdefault('MPLCONFIGDIR', str((out / 'matplotlib-cache').resolve()))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
        x = [r['wall_seconds'] / 60 for r in rows]
        axes[0].plot(x, [r['neural_seconds'] - rows[0]['neural_seconds'] for r in rows])
        axes[0].set_ylabel('Neural seconds')
        for k in ['health', 'ammo']:
            axes[1].plot(x, [r[k] for r in rows], label=k, linewidth=1)
        axes[1].legend(); axes[1].set_ylabel('Game counters')
        axes[2].plot(x, [r['turn'] for r in rows], label='turn')
        axes[2].plot(x, [r['forward'] for r in rows], label='move', alpha=.7)
        axes[2].legend(); axes[2].set_ylabel('BCI output')
        axes[3].plot(x, [r['population_hz'] for r in rows], linewidth=1)
        axes[3].set_ylabel('Mean spikes/neuron/s'); axes[3].set_xlabel('Observed wall minutes')
        title = f'{wall / 60:.1f}-minute fly-connectome Doom observation — no learning mechanism'
        if completion.get('not_a_complete_hour'):
            title = 'Analyzer preflight: partial recorded segment — not the final hour'
        fig.suptitle(title)
        fig.savefig(out / 'observation.png', dpi=160); fig.savefig(out / 'observation.pdf'); plt.close(fig)
        if full:
            fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
            ax.bar([str(e['episode']) for e in full], [e['kills'] for e in full])
            ax.set(xlabel='Complete episode (one ongoing neural run)', ylabel='Kills',
                title='Complete episode outcomes — descriptive, not a learning curve')
            fig.savefig(out / 'episodes.png', dpi=160); plt.close(fig)
    except ImportError:
        (out / 'plot-unavailable.txt').write_text('matplotlib is unavailable; CSV and JSON analysis are complete.\n')
    print(json.dumps({'out': str(out), 'snapshots': len(rows), 'events': len(events),
        'violations': dict(violations), 'neural_seconds': neural, 'complete_episodes': len(full)}, indent=2))

if __name__ == '__main__':
    main()
