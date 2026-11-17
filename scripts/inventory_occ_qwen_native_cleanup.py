"""Read-only inventory; no deletion, no GPU or checkpoint deserialization."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 start=time.perf_counter();dirs=[p for p in (ROOT/'artifacts').iterdir() if p.is_dir() and (p.name.startswith('native-occ-cache-') or p.name in ['occ-qwen-native-engineering-test','occ-qwen-native-pretrained-resume-test','occ-qwen-native-training'])]
 textfiles=[p for base in ['scripts','configs'] for p in (ROOT/base).glob('*') if p.is_file() and p.suffix in ['.py','.ps1','.json'] and p.name!=Path(__file__).name]
 texts=[(str(p.relative_to(ROOT)),p.read_text(encoding='utf-8-sig',errors='replace').replace('\\','/')) for p in textfiles];files=[]
 for d in dirs:
  for p in d.rglob('*'):
   if not p.is_file() or p.suffix not in ['.ckpt','.pt','.pth']:continue
   rel=str(p.relative_to(ROOT)).replace('\\','/');protected=p.parent.name=='occ-qwen-native-training' and p.name in ['best.ckpt','latest.ckpt']
   refs=[name for name,txt in texts if rel in txt or (d.name in txt and p.name in txt)]
   category='protected_current_valid_weight' if protected else ('historical_training_rollback_nonidentical_until_proven' if p.name=='pre-optimization.ckpt' else 'engineering_test_artifact')
   files.append({'path':str(p),'relative':rel,'bytes':p.stat().st_size,'sha256':sha(p),'protected':protected,'category':category,'script_config_references':refs,'recoverability':'No exact replacement claimed unless listed in byte-identical group. Scripts can rerun tests but may not reproduce original bytes.'})
 groups={}
 for f in files:groups.setdefault((f['bytes'],f['sha256']),[]).append(f)
 duplicates=[]
 for (size,h),rows in groups.items():
  if len(rows)>1:
   keep=next((x for x in rows if x['protected']),rows[0]);duplicates.append({'bytes_each':size,'sha256':h,'retain_path':keep['path'],'redundant_identical_paths':[x['path'] for x in rows if x is not keep],'recoverable_redundant_bytes':size*(len(rows)-1)})
 summaries=[]
 for d in dirs:
  rows=[x for x in files if Path(x['path']).is_relative_to(d)]
  if rows:summaries.append({'directory':str(d),'weight_or_test_tensor_bytes':sum(x['bytes'] for x in rows),'unprotected_candidate_bytes':sum(x['bytes'] for x in rows if not x['protected']),'file_count':len(rows)})
 report={'read_only':True,'deleted_files':[],'files':files,'byte_identical_groups':duplicates,'directories':summaries,'unprotected_candidate_bytes':sum(x['bytes'] for x in files if not x['protected']),'safe_duplicate_only_bytes':sum(x['recoverable_redundant_bytes'] for x in duplicates),'protected_outside_inventory':['all data and caches including data/native_occ_aligned_cache and v2 caches','models/Qwen3-VL-2B-Instruct','all docs/logs/reports/configs/scripts','other version weights not explicitly inventoried'],'caution':'Candidate != authorized irreversible deletion. Preserve one byte-identical copy before removing duplicates. pre-optimization is not equivalent to final best/latest. No source data/cache directory is proposed for recursive deletion.','seconds':time.perf_counter()-start}
 out=ROOT/'artifacts/occ-qwen-native-final-audit/cleanup-inventory.json';out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['files']},indent=2))
if __name__=='__main__':main()
