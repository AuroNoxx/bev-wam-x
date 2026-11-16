"""Pure validation helpers for bounded CARLA recordings."""
def validate_ticks(rows,dt=.05):
 if len(rows)<2:raise ValueError('at least two snapshots required')
 for a,b in zip(rows,rows[1:]):
  if b['frame']!=a['frame']+1:raise ValueError('missing or duplicate simulator frame')
  if abs(b['timestamp_s']-a['timestamp_s']-dt)>1e-4:raise ValueError('simulator delta mismatch')
 return True

def event_coverage(expected_frames,observed_frames,sensor_started,sensor_ended,transport_errors):
 # Accounting only. Caller must establish sensor semantics independently.
 complete=bool(expected_frames) and set(expected_frames)==set(observed_frames) and sensor_started is True and sensor_ended is True and not transport_errors
 return {'recording_complete':complete,'expected_count':len(set(expected_frames)),'observed_count':len(set(observed_frames)),'missing_frames':sorted(set(expected_frames)-set(observed_frames)),'sensor_semantics_certified':False,'driving_pass':None}
