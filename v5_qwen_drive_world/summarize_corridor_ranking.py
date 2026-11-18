"""Threshold and ranking diagnostics on supported cells in the forward 0-80 m corridor."""
import json
from pathlib import Path
import numpy as np


def main():
    folder = Path(__file__).resolve().parents[1]/'artifacts/v5-qwen-drive-world/subset2000/final-window-audit'
    with np.load(folder/'front-corridor-hist.npz') as payload:
        hist = payload['counts']
        bins = int(payload['bins'])
    recorded = json.loads((folder/'world-probe.json').read_text())
    results = []
    for h, methods in enumerate(hist,1):
        samples = [row for row in recorded if row['h'] == h]
        totals = [sum(row[region+'_'+kind] for row in samples for region in ('ahead_0_40','ahead_40_80'))
                  for kind in ('valid','positives')]
        for method, (neg,pos) in zip(('world','static'), methods):
            assert (neg.sum()+pos.sum(),pos.sum()) == tuple(totals)
            tp = np.cumsum(pos[::-1]); candidates = np.cumsum((neg+pos)[::-1])
            precision = np.divide(tp,candidates,out=np.zeros(bins),where=candidates>0)
            recall = tp/pos.sum()
            ap = float(np.sum(precision*pos[::-1])/pos.sum())
            entry = dict(horizon=h,method=method,valid=totals[0],positives=totals[1],
                         prevalence=totals[1]/totals[0],approx_ap=ap,thresholds={})
            for threshold in (.5,.25,.125):
                index=bins-1-int(threshold*bins)
                entry['thresholds'][threshold]=dict(tp=int(tp[index]),reported=int(candidates[index]),
                                                    recall=float(recall[index]),
                                                    precision=float(precision[index]) if candidates[index] else None)
            results.append(entry)
    (folder/'front-corridor-ranking.json').write_text(json.dumps(results,indent=2))
    for row in results:
        print(row['horizon'],row['method'],round(row['prevalence'],4),round(row['approx_ap'],4),row['thresholds'])


if __name__=='__main__':
    main()
