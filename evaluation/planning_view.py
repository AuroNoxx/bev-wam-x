"""Render the recorded decision bank, selected path and actual execution together."""
import numpy as np
from mini_wa_jepa.geometry import ego_to_pixels, world_to_ego
from mini_wa_jepa.visualization.rerun_logger import colorize_bev_4ch
from planning.candidates import TERM_NAMES


COLORS=('#28b4ff','#ff6464','#64dc64','#ffc828','#c864ff','#ff9632','#32ffc8','#bdbdbd',
        '#e040fb','#00e676','#ff1744','#651fff','#00b0ff','#f50057','#ffd600','#76ff03')


def decision_figure(data,index,chinese=False):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    bev=data['bev'][index];ax.imshow(colorize_bev_4ch(bev))
    selected=int(data['selected_index'][index]);bank=data['candidates'][index]
    for k,path in enumerate(bank):
        xy=ego_to_pixels(path[:,:2],128,64);name='MSE' if len(bank)==1 else chr(65+k)
        ax.plot(xy[:,0],xy[:,1],'--o',lw=3 if k==selected else 1.2,markersize=2,
                color=COLORS[k%8],alpha=1 if k==selected else .65,label=name)
        ax.text(xy[-1,0],xy[-1,1],name,color=COLORS[k%8],fontsize=10)
    xy=ego_to_pixels(data['plans'][index,:,:2],128,64)
    ax.plot(xy[:,0],xy[:,1],color='cyan',lw=4,label='本轮执行' if chinese else 'Selected / STOP rule')
    state=data['states'][index]
    past=world_to_ego(data['states'][max(0,index-40):index+1,:2],state[:2],state[2])
    xy=ego_to_pixels(past,128,64);ax.plot(xy[:,0],xy[:,1],color='white',lw=3,label='实际已走过' if chinese else 'Actual past')
    actual=data['trajectory_target'][index][data['trajectory_valid'][index]]
    if len(actual):
        xy=ego_to_pixels(actual[:,:2],128,64)
        ax.plot(xy[:,0],xy[:,1],color='#ffbf00',lw=2,label='随后实际行驶' if chinese else 'Actual after replanning')
    ax.scatter([63.5],[63.5],marker='^',c='white',s=100,edgecolors='black',zorder=10)
    ax.set_xlim(5,122);ax.set_ylim(110,5);ax.set_xticks([]);ax.set_yticks([])
    title=(f'决策 {index} · {index*.1:.1f}秒 · 选择 {data["selected_ids"][index]}' if chinese else
           f'Decision {index} | {index*.1:.1f}s | selected {data["selected_ids"][index]}')
    ax.set_title(title);ax.legend(loc='upper left',fontsize=8)
    return fig


def score_rows(data,index):
    rows=[]
    for i in range(len(data['candidates'][index])):
        score=float(data['scores'][index,i]);prob=data['probability'][index,i]
        row=dict(候选='MSE' if len(data['candidates'][index])==1 else chr(65+i),
                 总成本=None if np.isnan(score) else round(score,5),
                 碰撞概率=None if np.isnan(prob[0]) else round(float(prob[0]),4),
                 越界概率=None if np.isnan(prob[1]) else round(float(prob[1]),4),
                 拒绝原因=str(data['rejections'][index,i]) or '可选')
        for name,value in zip(TERM_NAMES,data['terms'][index,i]):
            row[name]=None if np.isnan(value) else round(float(value),5)
        rows.append(row)
    return rows
