"""Launch the upstream CARLA GUI example with a local Windows font workaround."""
from pathlib import Path
import importlib.util
import os
import pygame

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / 'runtime/CARLA-0.9.15/WindowsNoEditor/PythonAPI/examples/manual_control.py'

# Some Windows font registry entries are non-string values. Do not edit the registry.
font = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'Fonts/consola.ttf'
if not font.exists():
    font = Path(pygame.__file__).parent / pygame.font.get_default_font()
pygame.font.get_fonts = lambda: ['courier', 'mono']
pygame.font.match_font = lambda *args, **kwargs: str(font)

# In synchronous mode each frame advances 0.05 simulation seconds. Pace the
# client at 20 Hz to avoid running the simulation 3x faster than real time.
_Clock = pygame.time.Clock


class RealtimeClock:
    def __init__(self):
        self.clock = _Clock()

    def tick_busy_loop(self, fps=0):
        return self.clock.tick(20)

    def __getattr__(self, name):
        return getattr(self.clock, name)


pygame.time.Clock = RealtimeClock
pygame.display.set_caption('Highway OCC Flow - CARLA Town04')

spec = importlib.util.spec_from_file_location('carla_upstream_manual_control', EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)
original_restart = example.World.restart


def restart_on_highway(self):
    original_restart(self)
    if self.map.name.endswith('/Town04'):
        # Spawn 16 was inspected via Town04's map API: road 35, lane -2,
        # four same-direction driving lanes. Re-select on map/version changes.
        transform = self.map.get_spawn_points()[16]
        self.player.set_target_velocity(example.carla.Vector3D())
        self.player.set_target_angular_velocity(example.carla.Vector3D())
        self.player.set_transform(transform)


example.World.restart = restart_on_highway
if __name__ == '__main__':
    example.main()
