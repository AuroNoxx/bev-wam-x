"""Separate delivery completeness from frozen Stage 6 scientific gates."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image

from configs.stage6 import Stage6Config
from mini_wa_jepa.provenance import provenance,write_json


def build(root):
    cfg=Stage6Config();root=Path(root)
    evaluation=json.loads((root/'test_eval/report.json').read_text())
    data=json.loads((root/'branches/manifest.json').read_text())
    cache=json.loads((root/'features/manifest.json').read_text())
    reports={};checks={};evidence=[]
    for seed in cfg.training_seeds:
        for kind in ('conditioned','unconditional'):
            name=f'{kind}_{seed}';folder=root/'runs'/name;run=json.loads((folder/'report.json').read_text());reports[name]=run
            rrd=list((folder/'samples').rglob('*.rrd'));png=list((folder/'samples').rglob('*.png'));npz=list((folder/'samples').rglob('*.npz'))
            checks[name]=bool(run['completed'] and not run['smoke'] and len(run['epochs'])==cfg.epochs
                and run['encoder_frozen'] and 15e6<=run['parameter_count']<=100e6
                and (folder/'best.pt').is_file() and (folder/'last.pt').is_file()
                and len(rrd)==18 and len(png)==18 and len(npz)==18 and list((folder/'tensorboard').glob('*tfevents*')))
            evidence.extend(rrd+png+npz)
    branches=[b for g in data['groups'] for b in g['branches']]
    checks['cohort']=data['completed'] and len(data['groups'])==192 and len(branches)==768
    checks['raw_identity_and_alignment']=data.get('raw_identity_and_alignment_audit_passed') is True
    checks['cache']=cache['completed'] and all(cache['splits'][s]['samples']==n for s,n in [('train',512),('val',128),('test',128)])
    checks['evaluation']=evaluation['completed']
    checks['source_snapshots']=len(list((root/'runs').glob('*/source_snapshot.tar.gz')))==4
    artifact_checks=[]
    for path in evidence:
        if path.suffix=='.png':
            with Image.open(path) as im:im.verify()
        elif path.suffix=='.npz':
            with np.load(path) as arrays:
                if arrays['prediction'].shape!=arrays['target'].shape or not np.isfinite(arrays['prediction']).all():
                    raise ValueError(f'Invalid raw prediction evidence {path}')
        elif path.suffix=='.rrd':
            result=subprocess.run([sys.executable,'-m','rerun','rrd','print',str(path)],
                capture_output=True,text=True,check=True,timeout=60)
            required=['/bev/candidates/keep/commanded','/bev/candidates/slow/actual',
                      '/latent/candidates/left/predicted','/latent/candidates/right/target']
            if any(key not in result.stdout for key in required):raise ValueError(f'Missing candidate streams {path}')
            path.with_suffix('.contents.txt').write_text(result.stdout)
        artifact_checks.append(dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    checks['evidence_readable']=True
    checks['resources']=all(max(e['peak_allocated_gib'] for e in r['epochs'])<r['hardware']['gpu']['total_memory_gib'] for r in reports.values())
    checks['raw_branch_hashes']=all(hashlib.sha256((root/'branches'/b['path']).read_bytes()).hexdigest()==b['sha256'] for b in branches)
    tests=json.loads((root/'tests_result.json').read_text()) if (root/'tests_result.json').exists() else {}
    checks['tests']=tests.get('passed') is True
    scientific=bool(evaluation['scientific_passed']);delivery=all(checks.values())
    result=dict(**provenance(cfg),stage=6,completed=True,delivery_passed=delivery,scientific_passed=scientific,
        passed=delivery and scientific,delivery_gates=checks,scientific_gates=evaluation['scientific_gates'],
        chapter_interaction_redesign='deferred until after Stage 6; not implemented',stage7_started=False)
    write_json(root/'acceptance.json',result)
    rows=[]
    for name,row in evaluation['models'].items():
        rows.append(dict(run_id=name,model='action_conditioned_world' if row['conditioned'] else 'unconditional_world',
            ablation='commanded_trajectory' if row['conditioned'] else 'zero_command_matched',split='test',seed=row['seed'],
            metrics={k:row[k] for k in ('latent_mse','delta_mse','retrieval_accuracy','wrong_command_mse')}))
    write_json(root/'metrics.json',dict(**provenance(cfg),visualization_schema_version=1,spec_version=2,stage=6,
        seeds=list(cfg.training_seeds),runs=rows))
    write_json(root/'delivery_checks.json',dict(passed=delivery,checks=checks,evidence_files=len(evidence)))
    write_json(root/'evidence_manifest.json',dict(stage=6,files=artifact_checks))
    lines=['# Stage 6 — 动作条件世界预测','',
        f'实验交付通过：{delivery}；科学门槛通过：{scientific}；总验收通过：{delivery and scientific}。',
        '阶段7未启动。Chapter交互改造仍按用户要求排在阶段6之后，本阶段没有增加框架或制作教学视频。','',
        '同一段历史输入，比较保持路线、左偏、右偏、减速四条实验指定命令及其真实执行后果。没有训练候选选择器，不能称为自主规划。',
        '固定阶段4编码器作为目标；两个训练种子各训练条件/无条件模型，完全相同初始化与样本顺序。未来观察为0.5/1/1.5/2秒，不能与阶段4的0.1–0.4秒误差直接比较。','',
        '| 模型 | 训练种子 | Latent MSE | 后果差分 MSE | 候选对应准确率 | 错配命令 MSE |',
        '|---|---|---|---|---|---|']
    for name,r in evaluation['models'].items():
        lines.append(f'| {name} | {r["seed"]} | {r["latent_mse"]:.6f} | {r["delta_mse"]:.6f} | {r["retrieval_accuracy"]:.1%} | {r["wrong_command_mse"]:.6f} |')
    lines+=['','## 配对结果与门槛','']
    for seed,pair in evaluation['paired'].items():
        lines.append(f'- 种子{seed}：潜变量误差改善{pair["latent_gain"]:.2%}，条件−无条件的种子簇bootstrap 95%区间{pair["latent"]["delta_ci95"]}；后果差分改善{pair["delta_gain"]:.2%}。门槛：{evaluation["scientific_gates"][seed]}。')
    noise=evaluation['replay_noise'];outcome=evaluation['actual_branch_outcomes']
    lines+=['','## 数据与限制','',
        '训练/验证/测试为128/32/32个场景起点，512/128/128条分支；按地图家族与种子隔离，测试只有8个独立种子，区间不代表大规模场景覆盖。',
        f'同命令重复执行的潜变量噪声MSE={noise["latent_mse"]}，真实候选后果差分MSE={noise["actual_candidate_delta_mse"]}，比值={noise["delta_to_noise_ratio"]}。',
        '前期001/002因严格复现检查中断并保留MRE；003在验证数值容差后主动中止，改为三个独立采集进程按地图分区。原始与合并数据均保留。',
        '首次分区也被交通车状态偏差拦下；读取原生源码后改用文档化的force_destroy=True，避免跨重置复用车辆。长历史探针的8个场景全部逐字节复现，正式采集恢复严格零像素差异检查。未修改旧阶段默认环境，也未声称修复旧桌面窗口路径。',
        f'测试命令跟踪ADE={outcome["command_tracking_ade_m"]:.4f}m、FDE={outcome["command_tracking_fde_m"]}m；这些是命令与实际执行的偏差，不是新轨迹预测头的ADE/FDE。',
        f'碰撞分支={outcome["collision_branches"]}/{outcome["branches"]}，越界分支={outcome["out_of_road_branches"]}/{outcome["branches"]}。没有测量新的自主闭环成功率。',
        '潜变量不是未来道路/车辆图像；候选对应检索只是后果预测的离线检验，不是部署的选路器。主图的实线是仿真实际结果，虚线是实验命令。',
        '冻结编码器能防止本阶段通过目标塌缩降低误差，但本实验只验证这个既有表征下的条件预测，不证明完整规划有效。',
        '任一科学门槛未过，都不得宣称阶段6可靠完成或自动进入阶段7；不根据测试结果重调门槛或追加训练。','',
        '## 证据','',
        '`test_eval/report.json`和`*_raw.npz`保存逐候选结果、置信区间、健康度及错配控制；`runs/`包含四次训练的TensorBoard、检查点、周期NPZ/PNG/RRD和源码快照；`branches/`保存命令、控制、状态及实际未来；`features/`记录冻结编码器与数据hash。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'summary.json',dict(delivery_passed=delivery,scientific_passed=scientific,models=evaluation['models'],
        paired=evaluation['paired'],replay_noise=noise,actual_branch_outcomes=outcome))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('artifacts/restart_stage6'))
    a=p.parse_args();r=build(a.root);print(json.dumps({k:r[k] for k in ('passed','delivery_passed','scientific_passed')}),flush=True)


if __name__=='__main__':main()
