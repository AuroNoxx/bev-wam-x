"""Read-only legacy log adapter; cannot repair missing instrumentation."""
import json,argparse
from pathlib import Path
from driving_metric_contract import evaluate_episode
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--out',required=True);a=p.parse_args();folder=Path(a.folder);out=Path(a.out);assert not out.exists()
 report=json.loads((folder/'report.json').read_text());rows=[json.loads(x) for x in (folder/'telemetry.jsonl').read_text().splitlines() if x.strip()];samples=[];planning=[]
 for r in rows:
  samples.append({'t_s':r['simulation_s'],'speed_mps':r['speed_kmh']/3.6})
  # Legacy planning_ms repeats across control ticks, use sensor-frame identity.
  if 'sensor_frame' in r and 'planning_ms' in r:planning.append({'id':str(r['sensor_frame']),'wall_ms':r['planning_ms']})
 ep={'schema':'driving-metrics-v1','status':report.get('status','unknown'),'samples':samples,'planning_events':planning,'coverage':{},'events':[],'scenario_fingerprint':None,'route_completed':None}
 result=evaluate_episode(ep);result['source_folder']=str(folder);result['legacy_reported_collision_callbacks']=len(report['collisions']) if isinstance(report.get('collisions'),list) else None;result['legacy_error']=report.get('error');result['adapter_limits']=['No completeness/coverage certification for event streams, so empty list cannot certify zero collisions.','Legacy gap/lead fields lack validated same-path bumper-gap semantics; TTC/headway unavailable.','Road s may reset/branch, no route-progress coordinate certification.','No complete scenario/map/weather/traffic-seed fingerprint; cannot pair policies.','Legacy velocity logged before tick but timestamp after tick; scalar derivatives are approximate; not full lateral comfort.'];out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
