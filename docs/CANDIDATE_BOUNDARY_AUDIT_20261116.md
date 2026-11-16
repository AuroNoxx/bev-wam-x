# Boundary-inclusive candidate audit

Only CPU reads of existing two-input/eight-candidate evidence; no training, checkpoint writes, test access or simulator.

Independent NumPy implementation appends t0=(x0,y0,heading0)=(0,0,0), forward velocity=(observed scalar speed,0), matching training convention. Actual lateral t0 velocity is unavailable, so this is not fully observed physical initial state. Full25 intervals use0.2s. Five synthetic tests pass: straight constant velocity, straight constant acceleration, normalized/physical encode-decode roundtrip, first-speed jump detection, left-positive axis.

Confirmed first-interval scalar acceleration in A1 is+29.451819m/s²; B3 is−21.538818m/s². Earlier auxiliary scalar acceleration report omitted this interval. The existing training kinematic_residual already INCLUDED t0, so this is not evidence that training lacked a boundary consistency penalty. Candidate output position/speed channels themselves disagree even after correct decoding.

A3 first chord velocity=(11.52295,−4.05568)m/s, while endpoint trapezoidal velocity=(2.44587,−0.27863)m/s; residual norm9.83155m/s. B0 first chord=(15.67250,1.97717) vs trapezoid=(8.49666,.05110), residual7.42984. A0/A2 first predicted positions move backwards/sideways despite positive forward endpoint velocities. All these are predicted values, not executed maneuvers.

Full max|scalar accel| for A0..A3:11.8474,29.4518,14.4158,11.6070; B0..B3:9.2775,9.2230,11.1956,21.5388m/s². Full max|scalar jerk|:A83.9803,126.9797,111.8850,70.1382; B42.5680,51.0609,81.5893,160.3935m/s³. Jerk differences are between acceleration midpoints .1,.3,...; no prehistory jerk implied.

Chord speed exceeding endpoint speeds flags inconsistency with interpolated endpoint velocity; it is not a proof of physical impossibility if unrestricted hidden acceleration is allowed. No collision/TTC/safety certification. Semantic source-causality and actual source timestamps remain unresolved.

Artifacts: artifacts/masked-teacher-155m-continue2048/postfit-evidence/boundary-inclusive-audit.json and boundary-crosscheck.json. Script:scripts/audit_candidate_boundary.py.
