"""Toy 1D metric plumbing fixture, NOT CARLA or driving validation."""
import json,hashlib
from pathlib import Path
from driving_metric_contract import evaluate_episode,paired_comparison
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'artifacts/driving-metric-contract/toy-follow';out.mkdir(exist_ok=False)
 scenario={'engine':'toy1d-metric-fixture-v1','dt_s':.1,'steps':120,'initial_ego_mps':10.,'lead_mps':8.,'bumper_gap_m':25.,'route_m':100.,'seed':0};fingerprint=hashlib.sha256(json.dumps(scenario,sort_keys=True).encode()).hexdigest();results=[]
 for policy in ['rule_follow','deliberately_unsafe_accelerate']:
  speed=10.;gap=25.;progress=0.;rows=[];events=[];contact=False
  for i in range(121):
   rows.append({'t_s':i*.1,'speed_mps':speed,'lead_valid':True,'lead_in_path':True,'bumper_gap_m':max(0,gap),'closing_speed_mps':speed-8.,'route_progress_m':progress})
   if gap<=0 and not contact:events.append({'id':f'collision-onset-{i}','kind':'collision','t_s':i*.1});contact=True
   if gap>0:contact=False
   if i==120:break
   acceleration=max(-4.,min(2.,(gap-(3.+1.8*speed))*.25-(speed-8.))) if policy=='rule_follow' else 2.
   next_speed=max(0,speed+acceleration*.1);distance=(speed+next_speed)*.5*.1;progress+=distance;gap+=8*.1-distance;speed=next_speed
  ep={'schema':'driving-metrics-v1','scenario_fingerprint':fingerprint,'status':'completed','coverage':{'collision':True},'events':events,'samples':rows,'route_completed':progress>=100,'planning_events':[],'fixture_only':True,'policy':policy};r=evaluate_episode(ep);r['fixture_only']=True;r['policy']=policy;results.append(r);(out/(policy+'.json')).write_text(json.dumps({'input':ep,'metrics':r},indent=2,allow_nan=False))
 assert results[0]['safety']['collision_events']['value']==0
 assert results[1]['safety']['collision_events']['value']>=1
 pair=paired_comparison(*results);(out/'comparison.json').write_text(json.dumps(pair,indent=2));print(json.dumps([{'policy':x['policy'],'collision_events':x['safety']['collision_events']['value'],'ttc':x['safety']['min_ttc'],'progress':x['efficiency']['route_progress_delta']} for x in results],indent=2))
if __name__=='__main__':main()
