"""Display live Stage 3.2 predictions and their synchronized CARLA chase camera."""
import argparse
import multiprocessing as mp
from pathlib import Path
import queue
from time import perf_counter

import cv2
import numpy as np

from .sr_viewer import SRViewer


class LiveCamera:
    """One latest BGRA image shared with the display, without queueing full-resolution frames."""

    def __init__(self, ctx, size):
        self.pixels = ctx.RawArray("B", size[0] * size[1] * 4)
        self.frame = ctx.RawValue("q", -1)
        self.lock = ctx.Lock()

    def publish(self, image):
        if self.lock.acquire(False):
            try:
                memoryview(self.pixels).cast("B")[:] = image.raw_data
                self.frame.value = image.frame
            finally:
                self.lock.release()

    def upload(self, texture, previous):
        with self.lock:
            if self.frame.value != previous:
                texture.write(self.pixels)
            return self.frame.value


def _simulate(args, frames, camera, stopped, messages):
    from .live_carla_demo import drive
    drive(args, frames, camera, stopped, messages.put)


def inset_size(width, height):
    w = max(1, min(round(width * .36), round(height * .48 * 16 / 9)))
    return w, max(1, round(w * 9 / 16)), round(min(width, height) * .018)


def run_demo(args):
    size = (960, 540)
    title = "BEV-WAM-X | 3D SR + CARLA"
    viewer = SRViewer(size=size, backend="auto" if args.headless else "gpu", window=not args.headless,
                      maximized=not args.headless)
    ctx = mp.get_context("spawn")
    frames = ctx.Queue(maxsize=2)
    messages = ctx.Queue()
    stopped = ctx.Event()
    args.camera_size = ((1920, 1080) if args.headless else viewer.glfw.get_framebuffer_size(viewer.window))
    camera = LiveCamera(ctx, args.camera_size)
    worker = ctx.Process(target=_simulate, args=(args, frames, camera, stopped, messages))
    try:
        if not args.headless:
            glfw = viewer.glfw
            camera_texture = viewer.ctx.texture(args.camera_size, 4)
            camera_texture.swizzle = "BGRA"
            sr_texture = viewer.ctx.texture(size, 4)
            viewer.resolved.release()
            viewer.resolved = viewer.ctx.framebuffer(sr_texture)

        def draw(texture, viewport):
            viewer.ctx.viewport = viewport
            texture.use()
            viewer.hud_vao.render(viewer.gl.TRIANGLE_STRIP, vertices=4)

        def present(show_sr=False):
            # One swap presents both layers. Keep the chase camera's aspect on resize.
            viewer.ctx.screen.use()
            viewer.ctx.screen.clear(.04, .04, .04)
            viewer.ctx.disable(viewer.gl.BLEND | viewer.gl.DEPTH_TEST)
            fw, fh = glfw.get_framebuffer_size(viewer.window)
            cw, ch = args.camera_size
            scale = min(fw / cw, fh / ch)
            w, h = int(cw * scale), int(ch * scale)
            x, y = (fw - w) // 2, (fh - h) // 2
            draw(camera_texture, (x, y, w, h))
            if show_sr:
                viewer.hud_program["flip_y"].value = True
                sw, sh, margin = inset_size(w, h)
                draw(sr_texture, (x + margin, y + margin, sw, sh))
                viewer.hud_program["flip_y"].value = False
            glfw.swap_buffers(viewer.window)

        def status(message):
            print(message, flush=True)
            if not args.headless:
                w, h = args.camera_size
                image = np.full((h, w, 3), 24, dtype=np.uint8)
                cv2.putText(image, message, (36, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (235, 235, 235), 2, cv2.LINE_AA)
                camera_texture.write(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA))
                present()

        if not args.headless:
            glfw.set_window_title(viewer.window, title)
        status("Starting live inference")
        worker.start()
        display_times = []
        previous_frame = -1
        have_sr = False
        while True:
            if not args.headless:
                glfw.poll_events()
                if glfw.window_should_close(viewer.window):
                    break
            try:
                status(messages.get_nowait())
            except queue.Empty:
                pass
            try:
                scene, chase, paths = frames.get(timeout=.002)
            except queue.Empty:
                if worker.exitcode is not None:
                    break
                scene = None
            if scene is not None and (not args.headless or paths):
                sr = viewer.render(scene, readback=bool(paths), present=False)
                have_sr = True
                if paths:
                    h, w = chase.shape[:2]
                    sw, sh, margin = inset_size(w, h)
                    chase[h-margin-sh:h-margin, margin:margin+sw] = cv2.resize(sr, (sw, sh))
                    encoded = cv2.imencode(".png", chase)[1].tobytes()
                    for path in paths:
                        path.write_bytes(encoded)
            if not args.headless:
                current = camera.upload(camera_texture, previous_frame)
                if current != previous_frame:
                    previous_frame = current
                    present(have_sr)
                    display_times.append(perf_counter())
                    if len(display_times) == 31:
                        periods = np.diff(display_times) * 1000
                        print(f"Presented {30000 / periods.sum():.1f} FPS, p95 {np.percentile(periods, 95):.0f} ms", flush=True)
                        display_times = display_times[-1:]
    finally:
        stopped.set()
        viewer.close()
        if worker.pid is not None:
            # Drain in-flight packets so the worker's Queue feeder can exit on Windows.
            while worker.is_alive():
                try:
                    frames.get(timeout=.05)
                except queue.Empty:
                    pass
            worker.join()
        frames.close()
        messages.close()
    if worker.exitcode:
        raise RuntimeError(f"Live simulation exited with code {worker.exitcode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path(".cache/stage32/run-dynamics-v1/latest.ckpt"))
    parser.add_argument("--carla-root", type=Path, default=Path("runtime/Carla-1.0-Win64-Shipping"))
    parser.add_argument("--map", default="Town12")
    parser.add_argument("--port", type=int, default=2460)
    parser.add_argument("--quality", choices=("Low", "Medium", "High", "Epic"), default="Epic")
    parser.add_argument("--npc", type=int, default=30)
    parser.add_argument("--cruise-highway-kmh", type=float, default=65.0)
    parser.add_argument("--cruise-urban-kmh", type=float, default=38.0)
    parser.add_argument("--wm-horizon", type=int, default=10)
    parser.add_argument("--wm-candidates", type=int, default=16)
    parser.add_argument("--max-ticks", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-frames", type=int, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("logs/stage32-live-demo"))
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
