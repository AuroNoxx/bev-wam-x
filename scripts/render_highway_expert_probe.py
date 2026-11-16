"""Create factual replays and control diagnostics from a finished traffic probe."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);args=p.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    diagnostics=[]
    for report in summary['results']:
        folder=args.run/report['case']['name']
        if report['status']!='completed':continue
        rows=[json.loads(x) for x in (folder/'telemetry.jsonl').read_text().splitlines()]
        lookup={r['frame']:r for r in rows}
        frames=[]
        # Retain all original 2 Hz PNGs; replay uses 1 Hz frames at 4x speed.
        for path in sorted((folder/'frames').glob('*.png'))[::2]:
            im=Image.open(path).convert('RGB');d=ImageDraw.Draw(im);r=lookup.get(int(path.stem))
            d.rectangle((0,0,640,65),fill='black')
            d.text((7,3),report['case']['name']+' | 4x replay | TM + rules',fill='white',font=font)
            if r:
                d.text((7,23),f"t={r['time']:.1f}s | speed={r['actors']['ego']['speed']*3.6:.1f}km/h | lane={r['actors']['ego']['lane']}",fill='white',font=font)
                d.text((7,43),'state='+r['phase']+' (command state, not proof of completion)',fill='yellow',font=font)
            frames.append(im.quantize(colors=128))
        if frames:frames[0].save(folder/'replay-4x.gif',save_all=True,append_images=frames[1:],duration=250,loop=0,optimize=False)
        velocity=np.array([r['actors']['ego']['speed'] for r in rows])
        accel=np.diff(velocity)/.05
        smoothed=np.convolve(accel,np.ones(11)/11,'valid')
        brake=np.array([r['control']['brake'] for r in rows])
        diag={'case':report['case']['name'],
              'speed_difference_accel_p95_mps2':float(np.percentile(abs(accel),95)),
              'speed_difference_accel_max_abs_mps2':float(max(abs(accel))),
              'min_accel_mps2':float(min(accel)),
              'smoothed_0p55s_accel_p95_abs_mps2':float(np.percentile(abs(smoothed),95)),
              'smoothed_0p55s_accel_max_abs_mps2':float(max(abs(smoothed))),
              'brake_active_transitions':int(np.count_nonzero(np.diff(brake>.1))),
              'lane_ids_visited':sorted(set(r['actors']['ego']['lane'] for r in rows)),
              'max_lateral_error_m':max(abs(r['actors']['ego']['lateral_error']) for r in rows)}
        if report['case']['maneuver']=='following':
            diag['following_speed_segments']=[]
            for lo,hi in [(15,20),(25,30),(30,35),(40,45),(45,50)]:
                subset=[r for r in rows if lo<=r['time']<hi]
                diag['following_speed_segments'].append({'interval_s':[lo,hi],
                    'ego_mean_kmh':float(np.mean([r['actors']['ego']['speed']*3.6 for r in subset])),
                    'lead_mean_kmh':float(np.mean([r['actors']['lead']['speed']*3.6 for r in subset]))})
        diagnostics.append(diag)
    (args.run/'control-diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    print(json.dumps(diagnostics))


if __name__=='__main__':main()
