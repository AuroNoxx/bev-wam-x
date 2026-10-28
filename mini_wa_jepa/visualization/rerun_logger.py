"""The shared Rerun bridge for collection, replay, training and evaluation."""
from pathlib import Path
import os
import socket
import subprocess
import time
from uuid import uuid4

import numpy as np
import rerun as rr


def colorize_bev_4ch(bev: np.ndarray, ego_marker: bool = True) -> np.ndarray:
    """Fuse CHW uint8 channels; three channels are supported for legacy review."""
    bev = np.asarray(bev)
    if bev.ndim != 3 or bev.shape[0] not in (3, 4) or bev.dtype != np.uint8:
        raise ValueError(f"Expected (3|4, H, W) uint8 BEV, got {bev.shape} {bev.dtype}")
    _, h, w = bev.shape
    rgb = np.full((h, w, 3), (20, 24, 32), dtype=np.uint8)
    for channel, color in enumerate([(50, 130, 85), (245, 175, 45), (235, 60, 80)]):
        rgb[bev[channel] > 0] = color
    if bev.shape[0] == 4:
        occluded = bev[3] == 0
        rgb[occluded] = (rgb[occluded].astype(np.float32) * 0.35).astype(np.uint8)
    if ego_marker:
        cy, cx = h // 2, w // 2
        rgb[max(0, cy - 3):cy + 4, max(0, cx - 2):cx + 3] = 255
    return rgb


class RerunLogger:
    """Own recordings explicitly, without changing Rerun's global recording.

    File and viewer use separate streams when both are requested, so attaching a
    file sink cannot replace the viewer sink. Empty trajectories clear old lines;
    None leaves that stream unchanged (used for delayed ground-truth backfill).
    SDK 0.22.1 keeps the accepted NumPy 1.26 simulator environment reproducible;
    the time/scalar adapter also supports the existing SDK 0.39 installation.
    """

    def __init__(self, app_id="mini-wa-jepa", spawn=True, save_path=None, recording_id=None, viewer_port=None):
        self.app_id = app_id
        self._streams = []
        self._closed = False
        recording_id = str(recording_id or uuid4())
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.exists():
                raise FileExistsError(f"Preserve existing recording: {save_path}")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            stream = self._new_stream(app_id, recording_id)
            stream.save(str(save_path))
            self._streams.append(stream)
        if spawn or not self._streams:
            stream = self._new_stream(app_id, recording_id + "-viewer")
            if spawn:
                # Another SDK version may already own the default Rerun port.
                if viewer_port is None:
                    with socket.socket() as port_socket:
                        port_socket.bind(('127.0.0.1', 0))
                        viewer_port = port_socket.getsockname()[1]
                self._spawn_viewer(stream, viewer_port)
            self._streams.append(stream)
        self._send_blueprint()

    @staticmethod
    def _spawn_viewer(stream, port):
        # SDK 0.22's spawn searches PATH, which can select a different virtualenv.
        # Run the bundled executable explicitly, without changing process PATH.
        import rerun_cli
        executable = Path(rerun_cli.__file__).parent / ('rerun.exe' if os.name == 'nt' else 'rerun')
        child_env = dict(os.environ, RERUN_APP_ONLY='true')
        process = subprocess.Popen([str(executable), '--port', str(port), '--expect-data-soon'],
                                   env=child_env, start_new_session=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f'Rerun viewer exited with code {process.returncode}')
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.1):
                    break
            except OSError:
                time.sleep(.05)
        else:
            process.terminate()
            process.wait(timeout=5)
            raise TimeoutError('Rerun viewer did not start within 15 seconds')
        if hasattr(rr, 'set_time'):
            rr.connect_grpc(f'rerun+http://127.0.0.1:{port}/proxy', recording=stream)
        else:
            rr.connect_tcp(f'127.0.0.1:{port}', recording=stream)

    @staticmethod
    def _new_stream(app_id, recording_id):
        # 0.22 constructs streams through new_recording; newer SDKs expose a constructor.
        if hasattr(rr, "set_time"):
            return rr.RecordingStream(app_id, recording_id=recording_id)
        return rr.new_recording(app_id, recording_id=recording_id)

    def _send_blueprint(self):
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="BEV + trajectories", origin="/bev", contents=[
                    "/bev/composite", "/bev/navigation", "/bev/planned_trajectory", "/bev/ground_truth_trajectory", "/bev/candidates/**"]),
                rrb.Spatial2DView(name="Camera (when available)", origin="/camera")),
            rrb.Horizontal(*[rrb.Spatial2DView(name=name, origin=f"/bev/channels/{name}")
                             for name in ("drivable", "boundaries", "vehicles", "visibility_mask")]),
            rrb.TimeSeriesView(name="Telemetry", origin="/telemetry"),
            rrb.Horizontal(rrb.TensorView(name="Predicted latent", origin="/latent/predicted"),
                           rrb.TensorView(name="True latent", origin="/latent/ground_truth"))))
        for stream in self._streams:
            stream.send_blueprint(blueprint)

    def _log(self, path, data, *, static=False):
        if self._closed:
            raise RuntimeError("Recording is closed")
        for stream in self._streams:
            stream.log(path, data, static=static, strict=True)

    def set_step(self, step, sim_time_s=None):
        for stream in self._streams:
            if hasattr(rr, "set_time"):
                stream.set_time("step", sequence=int(step))
                if sim_time_s is not None:
                    stream.set_time("sim_time", duration=float(sim_time_s))
            else:
                stream.set_time_sequence("step", int(step))
                if sim_time_s is not None:
                    stream.set_time_seconds("sim_time", float(sim_time_s))

    def log_bev(self, bev, extent_m=64.0):
        if extent_m <= 0:
            raise ValueError("BEV extent must be positive")
        self._log("bev/composite", rr.Image(colorize_bev_4ch(bev)))
        names = ("drivable", "boundaries", "vehicles", "visibility_mask")
        for i, name in enumerate(names):
            self._log(f"bev/channels/{name}", rr.Image(bev[i]) if i < len(bev) else rr.Clear(recursive=True))

    def log_camera(self, rgb, name="camera/native_rgb"):
        self._log(name, rr.Image(rgb) if rgb is not None else rr.Clear(recursive=True))

    def log_navigation(self, points, valid, *, source, size=128, extent_m=64.0):
        """Declared current route input, visually separate from a commanded plan."""
        points, valid = np.asarray(points), np.asarray(valid, bool)
        if (points.ndim != 2 or points.shape[1] != 2 or valid.shape != (len(points),)
                or not np.isfinite(points).all() or not source or extent_m <= 0 or size <= 0):
            raise ValueError('Navigation requires finite [K,2] points, mask and source')
        if valid.any():
            pixels = (size-1)/2-points[valid][:, [1, 0]]*(size/extent_m)
            self._log('bev/navigation', rr.LineStrips2D([pixels], colors=[180, 100, 255]))
        else:
            self._log('bev/navigation', rr.Clear(recursive=True))
        self.log_context(navigation_source=source, navigation_valid=valid.tolist())

    def log_metrics(self, metrics, prefix="metrics"):
        scalar_type = rr.Scalars if hasattr(rr, "Scalars") else rr.Scalar
        for name, value in metrics.items():
            if value is not None:
                self._log(f"{prefix}/{name}", scalar_type(float(value)))

    def log_telemetry(self, speed_m_s=None, steering=None, throttle_brake=None,
                      lateral_error_m=None, route_completion=None):
        self.log_metrics(dict(speed_m_s=speed_m_s, steering=steering,
                             throttle_brake=throttle_brake, lateral_error_m=lateral_error_m,
                             route_completion=route_completion), prefix="telemetry")

    def log_context(self, *, static=False, **context):
        self._log("context", rr.AnyValues(**context), static=static)

    def log_state(self, state):
        """Six recorded fields: world xy, heading, speed, world vx/vy (SI units)."""
        self.log_metrics(dict(zip(("world_x_m", "world_y_m", "heading_rad", "speed_m_s",
                                   "world_vx_m_s", "world_vy_m_s"), state)), prefix="ego")

    def log_latents(self, *, predicted, ground_truth):
        """Log actual tensors [horizon, tokens, channels], without invented projections."""
        predicted, ground_truth = np.asarray(predicted), np.asarray(ground_truth)
        if predicted.shape != ground_truth.shape or predicted.ndim != 3:
            raise ValueError("Latents must have matching [horizon, tokens, channels] shapes")
        for name, tensor in (("predicted", predicted), ("ground_truth", ground_truth)):
            self._log(f"latent/{name}", rr.Tensor(tensor, dim_names=["horizon", "token", "channel"]))

    def log_trajectories(self, planned=None, ground_truth=None, size=128, extent_m=64.0):
        """Ego x forward/y left in metres; accepts (K,2), legacy (K,3), v2 (K,4)."""
        if size <= 0 or extent_m <= 0:
            raise ValueError("BEV size and extent must be positive")
        for name, trajectory, color in (("planned", planned, [40, 180, 255]),
                                         ("ground_truth", ground_truth, [255, 200, 40])):
            if trajectory is None:
                continue
            pts = np.asarray(trajectory)
            if pts.size == 0:
                self._log(f"bev/{name}_trajectory", rr.Clear(recursive=True))
                continue
            if pts.ndim != 2 or pts.shape[1] not in (2, 3, 4) or not np.isfinite(pts).all():
                raise ValueError("Trajectory must be finite (K, 2|3|4)")
            pixels = (size - 1) / 2 - pts[:, [1, 0]] * (size / extent_m)
            self._log(f"bev/{name}_trajectory", rr.LineStrips2D([pixels], colors=color))

    def log_candidate(self, candidate_id, *, commanded, actual, predicted_latent, target_latent,
                      color=(40, 180, 255), size=128, extent_m=64.0):
        """Record all experiment branches at one optimizer/simulation time, without fake selection."""
        candidate_id=str(candidate_id)
        if not candidate_id or not all(c.isalnum() or c in '_-' for c in candidate_id):
            raise ValueError('Candidate ID must be a nonempty path-safe identifier')
        if size<=0 or extent_m<=0:raise ValueError('Positive BEV geometry required')
        for label,points in (('commanded',commanded),('actual',actual)):
            points=np.asarray(points)
            if points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all():
                raise ValueError('Candidate paths require finite [K,4] waypoints')
            path=f'bev/candidates/{candidate_id}/{label}'
            if len(points):
                pixels=(size-1)/2-points[:,[1,0]]*(size/extent_m)
                self._log(path,rr.LineStrips2D([pixels],colors=list(color)))
            else:self._log(path,rr.Clear(recursive=True))
        predicted_latent=np.asarray(predicted_latent)
        if predicted_latent.ndim!=3:
            raise ValueError('Candidate prediction must be [H,N,D]')
        if target_latent is None:
            if len(predicted_latent):
                self._log(f'latent/candidates/{candidate_id}/predicted',rr.Tensor(
                    predicted_latent,dim_names=['horizon','token','channel']))
            else:
                self._log(f'latent/candidates/{candidate_id}/predicted',rr.Clear(recursive=True))
            self._log(f'latent/candidates/{candidate_id}/target',rr.Clear(recursive=True))
            return
        target_latent=np.asarray(target_latent)
        if predicted_latent.shape!=target_latent.shape:
            raise ValueError('Candidate latents require matching [H,N,D] arrays')
        for label,value in (('predicted',predicted_latent),('target',target_latent)):
            if len(value):self._log(f'latent/candidates/{candidate_id}/{label}',rr.Tensor(value,dim_names=['horizon','token','channel']))

    def log_planning(self, *, candidates, selected_id, selected_trajectory, selection_source,
                     scores=None, terms=None, rejection_reasons=None, predicted=None,
                     executed_action=None):
        """Record a real candidate bank and explicit selector/fallback provenance.

        No counterfactual target is invented for unexecuted plans. Actual subsequent
        replanning outcomes are logged separately as actual trajectories/observations.
        """
        import json
        candidates=np.asarray(candidates)
        ids=['MSE'] if len(candidates)==1 else [chr(65+i) for i in range(len(candidates))]
        colors=[(40,180,255),(255,100,100),(100,220,100),(255,200,40),
                (200,100,255),(255,150,50),(50,255,200),(200,200,200)]
        for i,name in enumerate(ids):
            latent=np.empty((0,64,256),np.float32) if predicted is None else predicted[i]
            self.log_candidate(name,commanded=candidates[i],actual=np.empty((0,4)),
                predicted_latent=latent,target_latent=None,color=colors[i%len(colors)])
        self.log_trajectories(planned=selected_trajectory)
        self.log_context(selected_id=selected_id,selection_source=selection_source,candidate_ids=ids,
            score_json=json.dumps(None if scores is None else np.asarray(scores).tolist()),
            terms_json=json.dumps(None if terms is None else np.asarray(terms).tolist()),
            rejection_json=json.dumps(rejection_reasons),
            executed_action_json=json.dumps(None if executed_action is None else np.asarray(executed_action).tolist()))

    def close(self):
        if not self._closed:
            for stream in self._streams:
                stream.flush()
                stream.disconnect()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
