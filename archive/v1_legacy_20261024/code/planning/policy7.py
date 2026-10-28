"""Three matched policies; deployment inputs contain no future observations or route."""
from pathlib import Path
import json
import numpy as np
import torch
from configs.stage7 import Stage7Config
from mini_wa_jepa.normalization import Normalizer
from models.flow_matcher import ConditionalFlowMatcher
from models.risk_head import ConsequenceRisk
from planning.candidates import score_candidates, emergency_trajectory
from scripts.cache_stage7_risk_features import load_world
from scripts.evaluate_joint import load_joint
from training.flow_matching import file_sha256


class Stage7Policy:
    def __init__(self,arm,seed,root='artifacts/restart_stage7',risk_runs=None,flow_path=None):
        self.arm=arm;self.seed=seed;cfg=Stage7Config();root=Path(root);self.provenance={}
        if arm=='mse':
            self.base,_,saved=load_joint(cfg.stage4_checkpoint,'cuda');self.encoder=self.base.encoder
            self.normalizer=Normalizer(**saved['normalization'])
            self.parameter_count=sum(p.numel() for p in self.base.parameters())
            self.provenance['mse_checkpoint_sha256']=file_sha256(cfg.stage4_checkpoint)
            return
        self.world,_=load_world();self.encoder=self.world.base.encoder
        path=Path(flow_path) if flow_path else root/f'flow_{seed}'/'best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        self.normalizer=Normalizer(**json.loads((root/'flow_features_001/normalization.json').read_text()))
        if saved.get('model_arch',{}).get('model_type')=='transformer_kinematic':
            from models.flow_matcher_transformer import TransformerKinematicFlowMatcher
            self.flow=TransformerKinematicFlowMatcher(**saved['model_arch'],normalizer=self.normalizer).to('cuda').eval()
        else:
            arch={k:v for k,v in saved['model_arch'].items() if k!='time_emb_dim'}
            self.flow=ConditionalFlowMatcher(**arch).to('cuda').eval()
        self.flow.load_state_dict(saved['model'])
        self.provenance.update(flow_checkpoint_sha256=file_sha256(path),world_checkpoint_sha256=file_sha256(cfg.stage6_checkpoint))
        self.risk=None;self.thresholds=None
        if arm=='full':
            run=Path(risk_runs or root/'risk_runs')/f'world_{seed}'
            training=json.loads((run/'report.json').read_text())
            if not training['completed'] or training['smoke']:
                raise ValueError('Full planner requires completed formal risk training')
            saved=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            self.risk=ConsequenceRisk(**saved['arch']).to('cuda').eval();self.risk.load_state_dict(saved['model'])
            self.thresholds=json.loads((run/'validation.json').read_text())['thresholds']
            self.provenance.update(risk_checkpoint_sha256=file_sha256(run/'best.pt'),thresholds=self.thresholds)
        if hasattr(self.world, 'base') and hasattr(self.world.base, 'encoder') and hasattr(self.world, 'world'):
            active_world = sum(p.numel() for p in self.world.base.encoder.parameters()) + sum(p.numel() for p in self.world.world.parameters())
        else:
            active_world = sum(p.numel() for p in self.world.parameters())
        self.parameter_count = active_world + sum(p.numel() for p in self.flow.parameters())
        if self.risk is not None:self.parameter_count+=sum(p.numel() for p in self.risk.parameters())
        if not 15e6<=self.parameter_count<=100e6:raise ValueError(self.parameter_count)

    @torch.no_grad()
    def decide(self,history,ego,bev,bevs,states,length,width,noise_seed):
        motion=torch.from_numpy(ego).to('cuda')
        if self.arm=='mse':
            result=self.base.forward_encoded(history,motion)
            plan=self.normalizer.inverse_trajectory(result['trajectory'][0].cpu().numpy())
            return dict(plan=plan,candidates=plan[None],probability=None,terms=None,scores=None,
                selected_index=0,selected_id='MSE',selection_source='frozen_stage4_action_head',
                rejection_reasons=[[]],future=result['latent'].cpu().numpy(),future_horizons_s=[.1,.2,.3,.4])
        cfg=Stage7Config()
        if getattr(self.flow,'model_arch',{}).get('model_type')=='transformer_kinematic':
            from models.flow_matcher_transformer import get_physical_candidate_actions
            v_curr = float(states[-1][3]) if len(states) else float(ego[-1, 0] * 10.0)
            actions = get_physical_candidate_actions(v_curr)
            normalized=self.flow.sample(history,motion,num_steps=cfg.flow_euler_steps,num_candidates=len(actions),temperature=0.85,nominal_mode=True,seed=noise_seed,actions=actions)
            flow_cands=self.normalizer.inverse_trajectory(normalized[0].cpu().numpy())
            from planning.candidates import generate_lane_change_candidates
            lane_cands=generate_lane_change_candidates(v_curr)
            candidates=np.concatenate([flow_cands,lane_cands],axis=0)
        else:
            condition=self.flow.extract_condition(history,motion)
            normalized=self.flow.sample(condition,num_steps=cfg.flow_euler_steps,num_candidates=cfg.num_flow_candidates,seed=noise_seed)
            candidates=self.normalizer.inverse_trajectory(normalized[0].cpu().numpy())
        probability=None;future=None
        if self.risk is not None:
            command=torch.from_numpy(candidates).to('cuda')
            h=history.expand(len(candidates),-1,-1,-1);e=motion.expand(len(candidates),-1,-1)
            predicted=self.world(h,e,command);probability=self.risk(h,e,command,predicted).sigmoid().cpu().numpy()
            future=predicted.cpu().numpy()
        result=score_candidates(candidates,bev,bevs,states,length,width,probability,self.thresholds)
        index=result['selected_index'];plan=candidates[index] if index>=0 else emergency_trajectory(states[-1][3])
        return dict(**result,plan=plan,candidates=candidates,probability=probability,
            selected_id=chr(65+index) if index>=0 else 'STOP',
            selection_source='candidate_score' if index>=0 else 'rule_emergency_stop_all_rejected',
            future=future,future_horizons_s=[.5,1.,1.5,2.])
