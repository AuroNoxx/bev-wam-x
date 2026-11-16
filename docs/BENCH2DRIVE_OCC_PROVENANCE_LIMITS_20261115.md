# OCC source provenance: confirmed code, uncertified release path

Pinned dataset: rethinklab/Bench2Drive-V0.0.4 revision d84ae18cf20a88b9220df1bc15c073d8824b7ac1. The local upstream directory has only occ_label_gen.py, occ_online_carla.py and occ_vis.py, not a complete collector checkout.

The inspected online implementation receives current semantic LiDAR and boxes, appends current cache to a bounded history queue, and transforms past object points into current poses. This implementation differs from offline OccDatasetGen with21frames/key index10, which aggregates both sides of the key frame. The latter must not be used as evidence that the online release necessarily contains future fusion.

The online implementation initializes normal occupancy to0 and semantic to-1, writes occupied observations/fill to1, and returns occupancy2/semantic-1 on no-point path. It uses simulator semantic and bounding-box privileged information and geometric filling; no independent measured-free-space confidence mask was established. Therefore source0 must not be certified physically free.

Our cache255 denotes unknown/out-of-canvas from preprocessing. Compact maps occ!=1 to0 and compresses occupied semantic classes; warp fills outside with255. Cache values do not prove how upstream sem=-1 was serialized.

Official references under review:
- https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/occ_online_carla.py
- https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/tools/data_collect.py
- https://github.com/Thinklab-SJTU/Bench2Drive/blob/7ec25d1c9f7522d923ce5f3420986cef1cb2d956/docs/v004_update.md

Research inspection found no OCC call in that data_collect.py version. Actual released archive generation call path and revision remain unverified. Frame pairing anno i / OCC i-1 is our audited preprocessing alignment, not yet independently explained by a verified upstream saving caller. Thus temporal input isolation passes are not a source-level causal certification. Continue foundational/oracle-style diagnostics; do not claim deployable perception or source causality until provenance is established or source regenerated causally.
