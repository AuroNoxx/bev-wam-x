"""Observed-input Flow geometry/Full selector with explicit strict fallback."""
import numpy as np
import torch
from torch import nn
from planning.integrated import strict_score, curvature_braking
from planning.navigation_reference import navigation_reference


class IntegratedPolicy(nn.Module):
    def __init__(self, encoder, flow, world=None, risk=None, thresholds=None, shortlist=4):
        super().__init__()
        self.encoder = encoder; self.flow = flow; self.world = world; self.risk = risk
        self.thresholds = thresholds; self.shortlist = shortlist
        if (world is None) != (risk is None):
            raise ValueError('Full policy requires both a world and a matched risk head')

    @torch.no_grad()
    def decide(self, history, ego, bev, bevs, states, navigation, vehicle, permission, seed):
        device = history.device
        reference = navigation_reference(ego, navigation.points[None], navigation.valid[None], vehicle[None])
        candidates = self.flow.sample(history, torch.from_numpy(ego).to(device),
            torch.from_numpy(navigation.points[None]).to(device), torch.from_numpy(navigation.valid[None]).to(device),
            torch.from_numpy(vehicle[None]).to(device), torch.from_numpy(reference).to(device),
            maneuver_allowed=torch.from_numpy(permission[None]).to(device), candidates=8, steps=10, seed=seed)[0]
        raw = candidates.cpu().numpy()
        initial = strict_score(raw, bev, bevs, states, *vehicle[:2], navigation, maneuver_allowed=permission)
        survivors = np.flatnonzero(initial['eligible'])
        selected = survivors[np.argsort(initial['scores'][survivors])[:self.shortlist]]
        probability = np.zeros((8, 2), np.float32); predicted = None
        if self.world is not None and len(selected):
            command = candidates[selected]
            predicted_tensor = self.world(history.expand(len(selected), -1, -1, -1),
                torch.from_numpy(np.repeat(ego, len(selected), axis=0)).to(device), command)
            probability[selected] = torch.sigmoid(self.risk(history.expand(len(selected), -1, -1, -1),
                torch.from_numpy(np.repeat(ego, len(selected), axis=0)).to(device), command, predicted_tensor)).cpu().numpy()
            predicted = [np.empty((0, 64, 256), np.float32) for _ in range(8)]
            for i, value in zip(selected, predicted_tensor.cpu().numpy()):
                predicted[i] = value
        final = initial
        if self.world is not None:
            final = strict_score(raw, bev, bevs, states, *vehicle[:2], navigation,
                                 probability, self.thresholds, maneuver_allowed=permission)
            # Only assessed shortlist candidates can be selected in Full.
            for i in range(8):
                if i not in selected:
                    final['eligible'][i] = False
                    final['rejection_reasons'][i].append('not_in_world_shortlist')
            eligible = final['eligible']
            final['selected_index'] = int(np.argmin(np.where(eligible, final['scores'], np.inf))) if eligible.any() else -1
        index = final['selected_index']; fallback = index < 0
        plan = curvature_braking(float(ego[0, -1, 0]*10), float(ego[0, -1, 1])) if fallback else raw[index]
        fallback_checks = None
        if fallback:
            fallback_checks = strict_score(plan[None], bev, bevs, states, *vehicle[:2], navigation,
                                            maneuver_allowed=permission)
        assessed = np.zeros(8, bool)
        if self.world is not None:
            assessed[selected] = True
        return dict(plan=plan, candidates=raw, reference=reference[0], selected_index=index,
                    selected_id='STOP' if fallback else chr(65+index), fallback=fallback,
                    fallback_geometry_safe=None if fallback_checks is None else bool(fallback_checks['eligible'][0]),
                    selection_source='measured_curvature_braking' if fallback else ('learned_Flow_WAM_risk' if self.world is not None else 'learned_Flow_geometry'),
                    probability=np.where(assessed[:, None], probability, np.nan), risk_assessed=assessed,
                    predicted=predicted, world_shortlist=selected,
                    scores=final['scores'], terms=final['terms'], rejection_reasons=final['rejection_reasons'])
