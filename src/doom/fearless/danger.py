"""Auditable artificial Doom-state threat channel for FEARLESS.

This is deliberately *not* visual perception and never returns a game action.
It translates a small observer-state vocabulary into additive current for LC4
and LPLC2.  The normal BCI remains the only button decoder.
"""

from dataclasses import dataclass
import hashlib
import math

import numpy as np


ENEMY_NAMES = frozenset(("DoomFlyImp", "Zombieman"))
PROJECTILE_TOKENS = ("projectile", "ball", "fireball", "rocket", "plasma")
THREAT_AMPLITUDE = 20.0
FALSE_ALARM_BLOCK_TICKS = 7


def wrap_degrees(value):
    return (value + 180.0) % 360.0 - 180.0


def direction_from_relative_angle(angle):
    if angle < -20:
        return "left"
    if angle > 20:
        return "right"
    return "center"


def packet_from_scene(scene, previous_distances, damage_taken):
    """Return a direction/intensity packet from pre-action observer geometry.

    ``visible`` means within the player's forward 120-degree field in this
    obstacle-free arena; it is an observer-state proxy, not a pixel test.
    """
    player = scene["player"]
    px, py, heading = player["x"], player["y"], player["angle"]
    candidates, distances = [], {}
    for obj in scene["objects"]:
        name = str(obj["name"])
        dx, dy = obj["x"] - px, obj["y"] - py
        distance = float(math.hypot(dx, dy))
        angle = wrap_degrees(math.degrees(math.atan2(dy, dx)) - heading)
        if name in ENEMY_NAMES:
            distances[int(obj["id"])] = distance
            kinds = []
            if abs(angle) <= 60:
                kinds.append("front_enemy")
                if name == "Zombieman":
                    kinds.append("hitscan_enemy")
            prior = previous_distances.get(int(obj["id"]))
            if prior is not None and prior - distance >= 3.0:
                kinds.append("rapidly_closing")
            if distance <= 72:
                kinds.append("imminent_collision")
            if kinds:
                intensity = max(0.35, min(1.0, (300.0 - distance) / 240.0))
                if "rapidly_closing" in kinds:
                    intensity = max(intensity, .70)
                if "imminent_collision" in kinds:
                    intensity = 1.0
                # Zombiemen are immediate hitscan threats; encode priority only
                # as stronger threat current, never as a motor command.
                if "hitscan_enemy" in kinds:
                    intensity = min(1.0, intensity + .20)
                candidates.append((intensity, angle, kinds, distance, name))
        elif any(token in name.lower() for token in PROJECTILE_TOKENS):
            candidates.append((1.0, angle, ["incoming_projectile"], distance, name))
    if damage_taken > 0:
        # Damage has no precise source direction in the exposed state, so retain
        # the closest current threat direction or explicitly mark it centre.
        angle = candidates[0][1] if candidates else 0.0
        candidates.append((1.0, angle, ["taking_damage"], 0.0, "damage"))
    if not candidates:
        return {"active": False, "intensity": 0.0, "direction": "center",
                "relative_angle_degrees": 0.0, "event_types": [],
                "nearest_enemy_distance": None, "distances": distances}
    intensity, angle, kinds, distance, target_name = max(candidates, key=lambda row: row[0])
    return {"active": True, "intensity": round(float(intensity), 6),
            "direction": direction_from_relative_angle(angle),
            "relative_angle_degrees": round(float(angle), 6),
            "event_types": sorted({kind for row in candidates for kind in row[2]}),
            "selected_threat_type": target_name,
            "nearest_enemy_distance": round(float(min(
                (row[3] for row in candidates if row[3] > 0), default=0.0)), 6),
            "distances": distances}


def blank_packet():
    return {"active": False, "intensity": 0.0, "direction": "center",
            "relative_angle_degrees": 0.0, "event_types": [],
            "nearest_enemy_distance": None, "distances": {}}


class FalseAlarmScheduler:
    """Causal block permutation of packets, preserving each completed block."""
    def __init__(self, seed, block_ticks=FALSE_ALARM_BLOCK_TICKS):
        self.seed = int(seed); self.block_ticks = int(block_ticks); self.pending = []
        self.ready = []; self.block = 0

    def push(self, packet):
        self.pending.append(packet)
        if len(self.pending) == self.block_ticks:
            order = list(range(self.block_ticks))
            digest = hashlib.sha256(f"{self.seed}:{self.block}".encode()).digest()
            shift = 1 + digest[0] % (self.block_ticks - 1)
            order = order[shift:] + order[:shift]
            self.ready.extend(self.pending[index] for index in order)
            self.pending = []; self.block += 1
        return self.ready.pop(0) if self.ready else blank_packet()


def threat_drive(packet, populations, amplitude=THREAT_AMPLITUDE):
    """Create LC4/LPLC2 current only; no motor/readout indices are touched."""
    if not packet["active"]:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    direction = packet["direction"]
    sides = ("L", "R") if direction == "center" else (("R",) if direction == "left" else ("L",))
    indices = np.unique(np.concatenate([populations[name][side]
        for name in ("LC4", "LPLC2") for side in sides]))
    return indices, np.full(len(indices), amplitude * packet["intensity"], dtype=np.float32)


def sided_populations(target_manifest):
    result = {}
    for name in ("LC4", "LPLC2"):
        result[name] = {}
        for side in ("L", "R"):
            result[name][side] = np.asarray([row["graph_index"] for row in
                target_manifest["populations"][name]["neurons"] if row["soma_side"] == side], dtype=np.int64)
    return result
