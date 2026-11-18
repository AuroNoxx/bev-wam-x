"""CPU-only charts from saved evaluation records; never fabricate predictions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'artifacts/v5-qwen-drive-world';out=root/'quality-view';out.mkdir(exist_ok=True)
    rows=[json.loads(x) for x in (root/'overfit64/validation.jsonl').read_text().splitlines()]
    last=rows[-1];epochs=[r['epoch'] for r in rows];all_m=last['metrics']['all'];high_m=last['metrics']['80_120']
    status=json.loads((root/'subset2000/status.json').read_text())
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                         'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#ccd3df','axes.labelcolor':'#334155','text.color':'#14243a',
                         'xtick.color':'#526075','ytick.color':'#526075','figure.facecolor':'#f7f9fc',
                         'axes.facecolor':'white','savefig.facecolor':'#f7f9fc'})
    fig,axes=plt.subplots(2,2,figsize=(15.6,10.3));fig.subplots_adjust(left=.073,right=.97,bottom=.12,top=.79,hspace=.43,wspace=.23)
    fig.text(.073,.949,'64条过拟合测试：真实评估结果',fontsize=24,weight='bold')
    fig.text(.073,.907,'同一批训练样本 · 5秒轨迹 · 每5轮评估一次 · 全部指标越低越好，基线改善率除外',fontsize=12,color='#526075')
    delta=(all_m['persistence_brier']-all_m['world_brier'])/all_m['persistence_brier']*100
    fig.text(.073,.853,f"第30轮   平均轨迹误差 {all_m['ade_m']:.2f} m     末端误差 {all_m['fde_m']:.2f} m     World整体比静态基线差 {abs(delta):.2f}%",fontsize=14,weight='bold')
    colors={'all':'#2465c7','80_120':'#dc7824'};labels={'all':'全部64条','80_120':'高速19条（100–120 km/h）'}
    ax=axes[0,0]
    for group in colors:
        values=[r['metrics'][group]['ade_m'] for r in rows]
        ax.plot(epochs,values,'o-',lw=2,color=colors[group],label=labels[group])
        ax.annotate(f'{values[-1]:.2f} m',(epochs[-1],values[-1]),xytext=(-4,12),textcoords='offset points',ha='right',color=colors[group],weight='bold')
    ax.set(title='平均轨迹误差 ADE：下降，但仍有米级偏差',xlabel='训练轮数',ylabel='平均距离误差 / m',xticks=epochs,ylim=(0,16))
    ax.legend(fontsize=10,loc='upper right');ax.grid(alpha=.16)
    ax=axes[0,1]
    for group in colors:
        values=[100*(r['metrics'][group]['persistence_brier']-r['metrics'][group]['world_brier'])/r['metrics'][group]['persistence_brier'] for r in rows]
        ax.plot(epochs,values,'o-',lw=2,color=colors[group],label=labels[group])
        ax.annotate(f'{values[-1]:+.2f}%',(epochs[-1],values[-1]),xytext=(-4,9),textcoords='offset points',ha='right',color=colors[group],weight='bold')
    ax.axhline(0,color='#718096',ls='--',lw=1)
    ax.set(title='World相对静态基线：整体未稳定胜出',xlabel='训练轮数',ylabel='Brier改善率 / %（正值更好）',xticks=epochs,ylim=(-7,14))
    ax.legend(fontsize=10,loc='upper left');ax.grid(alpha=.16)
    ax=axes[1,0]
    for source,color,name in [('bench2drive',colors['all'],'Bench2Drive 45条'),('highway',colors['80_120'],'Town04高速19条')]:
        selected=[r for r in last['samples'] if r['source']==source]
        ax.scatter([r['speed_kmh'] for r in selected],[r['ade_m'] for r in selected],s=40,alpha=.75,c=color,label=name,edgecolors='white',linewidths=.5)
    ax.set(title='第30轮逐样本误差：高速样本仍较困难',xlabel='实际自车速度 / km/h',ylabel='该样本ADE / m',xlim=(-3,123),ylim=(0,None))
    ax.legend(fontsize=10,loc='upper left');ax.grid(alpha=.16)
    ax=axes[1,1]
    for group in colors:
        values=[r['metrics'][group]['jerk_abs_max'] for r in rows]
        ax.plot(epochs,values,'o-',lw=2,color=colors[group],label=labels[group])
    ax.axhline(8,color='#84939b',ls='--',lw=1,label='代价项阈值8（不是停止门槛）')
    ax.set(title='动作平滑性：Jerk仍然很高',xlabel='训练轮数',ylabel='每条轨迹最大Jerk的平均值 / m/s³',xticks=epochs,ylim=(0,150))
    ax.legend(fontsize=9,loc='center right');ax.grid(alpha=.16)
    fig.text(.073,.057,f"第二步正在训练：第{status['epoch']}/3轮，已完成{status['step']}次更新。尚无第二步独立验证结果。",fontsize=11,color='#526075')
    fig.text(.073,.027,'这些是训练集过拟合结果，不代表泛化或闭环驾驶能力。原日志没有保存逐点轨迹/占用预测，图中未重建或虚构模型画面。',fontsize=10,color='#526075')
    fig.savefig(out/'quality-overview.png',dpi=145);fig.savefig(out/'quality-overview.svg');plt.close(fig)
    summary={'evaluation_scope':'64 training samples, intentional overfit evaluation; not held-out or closed-loop',
             'epoch':last['epoch'],'metrics':last['metrics'],'world_all_vs_persistence_percent':delta,
             'subset2000_snapshot':{k:status[k] for k in ('epoch','step','state')},
             'predicted_trajectory_and_occupancy_arrays_available':False,
             'source':'artifacts/v5-qwen-drive-world/overfit64/validation.jsonl'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(out/'quality-overview.png')


if __name__=='__main__':main()
