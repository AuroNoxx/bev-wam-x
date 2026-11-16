"""Bounded empty-world synchronous health probe; no vehicle policy or safety claim."""
import json,time,sys,argparse
from pathlib import Path
import carla
from driving_metric_contract import scenario_fingerprint,evaluate_episode
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--out',default='artifacts/carla-metric-health-20261116');args=parser.parse_args();out=ROOT/args.out;out.mkdir(exist_ok=False);report={'status':'connecting','vehicles_spawned':False,'training':False,'cloud_llm':False};original=None;world=None
 deadline=time.monotonic()+120;client=carla.Client('127.0.0.1',2100);client.set_timeout(2)
 try:
  while True:
   try:world=client.get_world();break
   except RuntimeError:
    if time.monotonic()>deadline:raise
    time.sleep(2)
  if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):raise RuntimeError('Existing vehicles/sensors; refusing to take over')
  original=world.get_settings()
  if original.synchronous_mode:raise RuntimeError('Already synchronous; possible other client, refusing')
  settings=world.get_settings();settings.synchronous_mode=True;settings.fixed_delta_seconds=.05;settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10;client.set_timeout(10);world.apply_settings(settings)
  rows=[];start=time.perf_counter()
  for i in range(10):
   frame=world.tick(10);snap=world.get_snapshot();assert frame==snap.frame
   rows.append({'frame':frame,'timestamp_s':snap.timestamp.elapsed_seconds,'delta_s':snap.timestamp.delta_seconds})
  assert all(abs(b['timestamp_s']-a['timestamp_s']-.05)<1e-4 for a,b in zip(rows,rows[1:]))
  weather=world.get_weather();config={'simulator_version':client.get_server_version(),'map':world.get_map().name,'weather':str(weather),'traffic_seed':0,'route':{'type':'empty-world-health-no-route'},'initial_actors':[],'fixed_delta_seconds':.05,'sensor_config':{'none':True},'controller_version':'none-health-only'}
  fingerprint=scenario_fingerprint(config);ep={'schema':'driving-metrics-v1','scenario_fingerprint':fingerprint,'status':'completed','coverage':{},'events':[],'samples':[{'t_s':x['timestamp_s']} for x in rows]}
  report.update(status='completed',ticks=rows,wall_seconds=time.perf_counter()-start,scenario=config,scenario_fingerprint=fingerprint,metrics=evaluate_episode(ep),limits='Empty world ticks only; no collision sensors or actor, all driving/safety metrics unavailable')
 except Exception as e:report.update(status='failed',error=repr(e))
 finally:
  if original is not None and world is not None:
   try:world.apply_settings(original);report['settings_restored']=True
   except Exception as e:report['settings_restored']=False;report['restore_error']=repr(e)
  (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2),flush=True)
 return 0 if report['status']=='completed' else 1
if __name__=='__main__':sys.exit(main())
