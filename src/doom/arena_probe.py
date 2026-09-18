"""Arena-only fixed-controller negative control. Never used by the live brain.

Replays a constant turn/move/fire command to detect trivial survival loopholes.
No neural model, learning, reward, god mode or guaranteed-death timer is used.
The five-minute observation cap censors survivors; it does not kill the player.
"""
import argparse
import hashlib
import json
from pathlib import Path
import vizdoom as vzd


def probe(wad, seed, seconds=300):
    manifest = json.loads(Path(wad).with_suffix('.json').read_text())
    game = vzd.DoomGame()
    game.load_config(str(Path(__file__).parent/'scenarios/combat_survival.cfg'))
    game.set_doom_scenario_path(str(Path(wad).resolve()))
    game.set_doom_game_path(str(Path(vzd.__file__).parent/'freedoom2.wad'))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_seed(seed)
    game.set_available_buttons([vzd.Button.TURN_LEFT_RIGHT_DELTA,
        vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, vzd.Button.ATTACK])
    game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 6)
    game.set_button_max_value(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, 20)
    game.init()
    try:
        game.new_episode()
        start = game.get_episode_time()
        # Means from the observed v1 loop, declared before changing the arena.
        # Use the same one-tic action boundary as the live broadcaster.
        while not game.is_episode_finished() and game.get_episode_time()-start < 35*seconds:
            game.make_action([.772, 19.226, True], 1)
        result = {'seed': seed, 'game_seconds': (game.get_episode_time()-start)/35,
            'dead': game.is_player_dead(), 'censored': not game.is_player_dead(),
            **{name: int(game.get_game_variable(var)) for name, var in [
                ('health', vzd.GameVariable.HEALTH), ('kills', vzd.GameVariable.KILLCOUNT),
                ('enemies', vzd.GameVariable.USER1), ('imps', vzd.GameVariable.USER5),
                ('zombies', vzd.GameVariable.USER6)]}}
        if 'maximum_alive_per_type' not in manifest:
            result['imps'] = result['zombies'] = None  # Not measured in v1.
        return result
    finally:
        game.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wad', type=Path, default=Path(__file__).parent/'scenarios/combat_survival.wad')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seeds = list(range(41027, 41035))
    result = {'purpose': 'environment QA; fixed controller, not neural behavior or learning',
        'wad_sha256': hashlib.sha256(args.wad.read_bytes()).hexdigest(),
        'action': {'turn': .772, 'forward': 19.226, 'attack': True},
        'observation_cap_game_seconds': 300,
        'development_seeds': seeds[:3], 'held_out_seeds': seeds[3:],
        'runs': [probe(args.wad, seed) for seed in seeds]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
