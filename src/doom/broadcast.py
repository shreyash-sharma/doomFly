"""Bounded, immutable batches of actual frames for CDN fan-out.

Display batching never advances the game, changes inputs or interpolates frames.
Each frame retains its original capture time, neural readouts and audit hashes.
"""
import json
import threading
from collections import OrderedDict

DISPLAY_FPS = 8
PLAYOUT_DELAY_MS = 2500
RETAIN_SEGMENTS = 30
MAX_SEGMENT_BYTES = 3_000_000


def encode(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


class Broadcast:
    def __init__(self):
        self.lock = threading.Lock()
        self.segments = OrderedDict()
        self.pending = []
        self.bucket = None
        self.run_id = None
        self.shared = None
        self.index = encode({'status': 'starting'})
        self.state = encode({'status': 'starting', 'generated_at_ms': 0})

    def publish(self, data):
        if data['run_id'] != self.run_id:
            with self.lock:
                self.segments.clear()
                self.pending = []
                self.bucket = None
                self.run_id = data['run_id']
            self.shared = {k: data[k] for k in ['schema', 'status', 'run_id', 'condition', 'decoder', 'manifest', 'protocol', 'reward']}
            self.shared['retina'] = {k: data['retina'][k] for k in ['uv', 'full_sample_count', 'display_stride']}
            self.shared['raster_ids'] = data['raster']['neuron_ids']
        bucket = data['generated_at_ms'] // 1000
        if self.bucket is not None and bucket != self.bucket:
            packet = encode({'transport': 1, 'shared': self.shared, 'frames': self.pending})
            if len(packet) > MAX_SEGMENT_BYTES:
                raise ValueError('Broadcast segment exceeds its declared size bound')
            with self.lock:
                self.segments[str(self.bucket)] = packet
                while len(self.segments) > RETAIN_SEGMENTS:
                    self.segments.popitem(last=False)
            self.pending = []
        self.bucket = bucket
        frame = {k: data[k] for k in ['sequence', 'generated_at_ms', 'frame', 'clocks', 'game', 'episodes',
            'action', 'readouts', 'total_spikes', 'window_spikes', 'window_ms', 'total_action_ticks',
            'audit', 'reward']}
        frame['luminance'] = data['retina']['luminance']
        if 'learning' in data:frame['learning'] = data['learning']
        if 'spectator' in data:frame['spectator'] = data['spectator']
        frame['raster_bin'] = data['raster']['bins'][-1]
        self.pending.append(frame)
        # Serialize once per captured frame, never once per viewer.
        encoded_state = encode(data)
        with self.lock:
            self.state = encoded_state
            self.index = encode({'transport': 1, 'status': 'running', 'run_id': self.run_id,
                'generated_at_ms': data['generated_at_ms'], 'sequence': data['sequence'],
                'segments': list(self.segments)[-4:], 'playout_delay_ms': PLAYOUT_DELAY_MS,
                'capture_fps_limit': DISPLAY_FPS, 'phase': data['protocol'].get('phase','baseline')})

    def offline(self, data):
        with self.lock:
            self.state = encode(data)
            self.index = self.state

    def get_segment(self, run_id, segment):
        with self.lock:
            return self.segments.get(segment) if run_id == self.run_id else None
