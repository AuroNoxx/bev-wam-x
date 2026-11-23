"""Causal, motion-compensated LiDAR sweeps for asynchronous inference."""
import numpy as np

from .geometry import local


def full_revolution(packets, reference_pose, reference_time, *, frequency=10., channels=64):
    """Assemble chronological (measurement, tick_duration) pairs.

    CARLA emits a partial sweep each tick. Return (FLU XYZI, diagnostics), or
    None until one complete causal revolution is available. Compensation uses
    each packet's sensor pose; it does not deskew moving objects or individual
    rays. The oldest packet is trimmed by return order per laser channel, an
    approximation because CARLA does not provide per-return timestamps.
    No simulator stepping, ground-truth objects, or disk I/O is involved.
    """
    if frequency <= 0 or channels <= 0:
        raise ValueError("Positive rotation frequency and channel count required")
    period = 1. / frequency
    candidates = [(p, float(dt)) for p, dt in packets if p.timestamp <= reference_time + 1e-6]
    if not candidates or reference_time - candidates[-1][0].timestamp > 1e-3:
        return None
    remaining, pieces, count = period, [], 0
    end = reference_time
    for packet, dt in reversed(candidates):
        if dt <= 0 or dt > period + 1e-4 or abs(end - packet.timestamp) > 1e-3:
            return None
        points = np.frombuffer(packet.raw_data, '<f4').reshape(-1, 4)
        take = min(dt, remaining)
        if take < dt - 1e-6:
            # Trim time within each channel, not channels from the flat buffer.
            parts, offset = [], 0
            for channel in range(channels):
                n = packet.get_point_count(channel)
                keep = int(round(n * take / dt))
                parts.append(points[offset + n - keep:offset + n])
                offset += n
            if offset != len(points):
                raise ValueError("LiDAR channel counts do not match packet length")
            points = np.concatenate(parts)
        scan = points.copy()
        sensor = np.asarray(packet.transform.get_matrix(), dtype=np.float64)
        scan[:, :3] = local(scan[:, :3] @ sensor[:3, :3].T + sensor[:3, 3], reference_pose)
        pieces.append(scan)
        remaining -= take
        count += 1
        if remaining <= 1e-6:
            sweep = np.concatenate(pieces[::-1])
            azimuth = np.arctan2(sweep[:, 1], sweep[:, 0])
            sectors = np.floor((azimuth + np.pi) * (72 / (2 * np.pi))).astype(int) % 72
            covered = int(np.count_nonzero(np.bincount(sectors, minlength=72)))
            return sweep, dict(packets=count, occupied_sectors=covered, span_seconds=period,
                               motion_compensation="per_packet")
        end = packet.timestamp - dt
    return None
