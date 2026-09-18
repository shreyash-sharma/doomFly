"""Read-only, bounded observation of the existing broadcaster; never starts a brain.

Captures complete telemetry every five wall seconds, unmodified input-frame JPEGs
every thirty seconds, every new local audit event, and a public-feed check each
minute. It does not add learning, reset episodes, or change model parameters.
"""
import argparse, base64, gzip, hashlib, json, os, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = 'http://127.0.0.1:8766/state'
PUBLIC = 'https://doomfly.example/api/live'

def stamp():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()

def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'Cache-Control': 'no-cache'}), timeout=8) as r:
        return json.load(r)

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=3600)
    args = p.parse_args()
    if args.seconds <= 0:
        p.error('Duration must be positive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    first = fetch(LOCAL)
    if first.get('status') != 'running':
        raise RuntimeError('The existing broadcaster is not running')
    # Check source provenance without loading another full neural simulation.
    files = {name: sha(ROOT / name) for name in first['provenance']['source_sha256']}
    files['outputs/doom/malecns_v1/graph.npz'] = sha(ROOT / 'outputs/doom/malecns_v1/graph.npz')
    save(out / 'fingerprints-start.json', files)
    started = time.time()
    began = time.monotonic()
    deadline = began + args.seconds
    protocol = {
        'started_at': stamp(), 'started_epoch': started,
        'scheduled_end_epoch': started + args.seconds, 'requested_wall_seconds': args.seconds,
        'run_id': first['run_id'], 'initial_sequence': first['sequence'],
        'initial_game': first['game'], 'initial_clocks': first['clocks'],
        'condition': first['condition'], 'decoder': first['decoder'], 'reward': first['reward'],
        'snapshot_interval_seconds': 5, 'frame_interval_seconds': 30,
        'public_check_interval_seconds': 60, 'audit_capture': 'all appended per-tic events',
        'existing_run_continues': True, 'neural_state_not_reset': True,
        'frozen_configuration': True, 'observation_only': True,
        'planned_checks': ['clock alignment', 'neural-to-action decoder reconstruction',
            'frame digest', 'finite/bounded telemetry', 'input/raster/population consistency',
            'audit continuity', 'complete episode outcomes versus censored episodes',
            'early/late descriptive activity and game performance', 'public-feed availability',
            'source and graph stability', 'no plasticity or learning claims'],
        'limits': ['One ongoing reconstructed male, no randomized intervention in this hour.',
            'Read-only telemetry is not a restartable full membrane/synapse checkpoint.',
            'No access to live weight arrays; disk hashes and reviewed source are separate evidence.',
            'An hour of wall time is not an hour of neural time.',
            'Performance trends alone cannot demonstrate learning or biological validity.'],
    }
    save(out / 'protocol.json', protocol)
    save(out / 'initial-state.json', first)
    log_path = ROOT / 'outputs/doom/audit.jsonl'
    audit = log_path.open()
    audit.seek(0, os.SEEK_END)
    inode = os.fstat(audit.fileno()).st_ino
    counts = {'snapshots': 0, 'frames': 0, 'audit_events': 0, 'public_checks': 0, 'errors': 0}
    next_frame = next_public = next_sample = began
    with gzip.open(out / 'snapshots.jsonl.gz', 'wt', compresslevel=3) as states, \
         gzip.open(out / 'events.jsonl.gz', 'wt', compresslevel=3) as events, \
         (out / 'public-checks.jsonl').open('w') as public, \
         (out / 'errors.jsonl').open('w') as errors:
        def error(stage, e):
            counts['errors'] += 1
            errors.write(json.dumps({'at': stamp(), 'stage': stage, 'error': str(e)}) + '\n')
            errors.flush()

        def drain_audit():
            nonlocal audit, inode
            for _ in range(2):
                while True:
                    pos = audit.tell()
                    line = audit.readline()
                    if not line:
                        break
                    if not line.endswith('\n'):
                        audit.seek(pos)
                        break
                    event = json.loads(line)
                    events.write(json.dumps({'received_at': stamp(), 'event': event}, separators=(',', ':')) + '\n')
                    counts['audit_events'] += 1
                current_inode = log_path.stat().st_ino
                if current_inode == inode:
                    break
                audit.close()
                audit = log_path.open()
                inode = current_inode
            events.flush()

        while True:
            now = time.monotonic()
            final_sample = now >= deadline
            elapsed = now - began
            try:
                drain_audit()
                state = fetch(LOCAL)
                if state.get('status') != 'running':
                    raise RuntimeError(f"Broadcaster status: {state.get('status')}")
                jpeg = base64.b64decode(state.pop('frame').split(',', 1)[1], validate=True)
                digest = hashlib.sha256(jpeg).hexdigest()
                if now >= next_frame or final_sample:
                    name = f"{counts['frames']:04d}-{state['sequence']}.jpg"
                    (out / 'frames' / name).write_bytes(jpeg)
                    state['saved_frame'] = name
                    counts['frames'] += 1
                    next_frame = now + 30
                record = {'received_at': stamp(), 'received_epoch': time.time(),
                    'elapsed_wall_seconds': elapsed, 'jpeg_digest_valid': digest == state['display_jpeg_sha256'],
                    'state': state}
                states.write(json.dumps(record, separators=(',', ':')) + '\n')
                states.flush()
                counts['snapshots'] += 1
                save(out / 'last-state.json', record)
            except Exception as e:
                error('local_sample', e)
            if now >= next_public or final_sample:
                try:
                    requested = time.monotonic()
                    state = fetch(PUBLIC)
                    public.write(json.dumps({'received_at': stamp(), 'received_epoch': time.time(),
                        'elapsed_wall_seconds': time.monotonic() - began,
                        'request_seconds': time.monotonic() - requested,
                        'state': {k: state.get(k) for k in ['status', 'run_id', 'sequence', 'generated_at_ms',
                            'condition', 'decoder', 'clocks', 'game', 'action', 'reward', 'provenance']}}, separators=(',', ':')) + '\n')
                    public.flush()
                    counts['public_checks'] += 1
                except Exception as e:
                    error('public_sample', e)
                next_public = now + 60
            save(out / 'progress.json', {'status': 'recorded' if final_sample else 'recording',
                'updated_at': stamp(), 'elapsed_wall_seconds': time.monotonic() - began,
                'scheduled_end_epoch': started + args.seconds, **counts})
            if final_sample:
                drain_audit()
                break
            next_sample += 5
            if next_sample < time.monotonic():
                next_sample = time.monotonic() + 5
            time.sleep(max(0, min(next_sample, deadline) - time.monotonic()))
    audit.close()
    save(out / 'fingerprints-end.json', {name: sha(ROOT / name) for name in files})
    save(out / 'complete.json', {'status': 'complete', 'completed_at': stamp(),
        'observed_wall_seconds': time.monotonic() - began, **counts})
    print(json.dumps({'status': 'complete', 'out': str(out), **counts}), flush=True)

if __name__ == '__main__':
    main()
