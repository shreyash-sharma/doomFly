"""Independent public HTTP probe using curl's transport; same fixed end time.

The primary collector's urllib transport received HTTP 403 while curl received
HTTP 200. Preserve both records; do not count client-specific rejection as a
confirmed visitor outage or silently erase it.
"""
import argparse, json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.out / 'protocol.json').read_text())
    end = protocol['scheduled_end_epoch']
    with (args.out / 'public-curl-checks.jsonl').open('x') as output:
        while True:
            began = time.monotonic()
            record = {'received_at': datetime.now(timezone.utc).isoformat(),
                      'received_epoch': time.time(), 'transport': 'curl',
                      'elapsed_wall_seconds': time.time() - protocol['started_epoch']}
            try:
                result = subprocess.run(['curl', '--fail', '--silent', '--show-error', '--max-time', '8',
                    'https://doomfly.example/api/live'], capture_output=True, text=True, check=True)
                state = json.loads(result.stdout)
                record['state'] = {k: state.get(k) for k in ['status', 'run_id', 'sequence', 'generated_at_ms',
                    'condition', 'decoder', 'clocks', 'game', 'action', 'reward', 'provenance']}
                record['ok'] = state.get('status') == 'running'
            except Exception as e:
                record.update(ok=False, error=str(e))
            record['request_seconds'] = time.monotonic() - began
            output.write(json.dumps(record, separators=(',', ':')) + '\n')
            output.flush()
            if time.time() >= end:
                break
            time.sleep(min(60, max(0, end - time.time())))

if __name__ == '__main__':
    main()
