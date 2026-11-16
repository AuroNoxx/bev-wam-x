"""Sparse image/semantic projection check; no model inputs changed."""
from pathlib import Path
import gzip,json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/diagnostics/20261115-contract'
def main():
    rows=[];panels=[]
    for file in sorted((OUT/'rgb-probe').rglob('semantic_front-*.png')):
        route=file.parent.name;i=int(file.stem.split('-')[-1]);clip=ROOT/'data/public/bench2drive/clips'/route
        a=json.load(gzip.open(clip/'anno'/f'{i:05}.json.gz','rt'));e=next(b for b in a['bounding_boxes'] if b['class']=='ego_vehicle');cam=a['sensors']['CAM_FRONT'];rel=np.asarray(cam['world2cam'])@np.linalg.inv(e['world2ego']);k=np.asarray(cam['intrinsic']);im=np.asarray(Image.open(file));rgb=Image.open(file.with_name(f'rgb_front-{i:05}.jpg')).convert('RGB')
        for d in [-2,-1,0,1,2]:
            with np.load(clip/'occ'/f'{i+d:05}.npz') as f:
                mask=(f['occ']==1)&np.isin(f['seg'],[14,15,16,17,18,19]);y,x,z=np.nonzero(mask)
            p=np.stack([-25.6+(x+.5)*.4,-25.6+(y+.5)*.4,-5+(z+.5)*.4,np.ones(len(x))],1)@rel.T;p=p[p[:,0]>.5]
            u=np.rint(k[0,0]*p[:,1]/p[:,0]+k[0,2]).astype(int);v=np.rint(k[1,1]*(-p[:,2])/p[:,0]+k[1,2]).astype(int);ok=(u>=0)&(u<im.shape[1])&(v>=0)&(v<im.shape[0]);u,v=u[ok],v[ok]
            rows.append(dict(route=route,image_frame=i,occ_offset=d,projected_points=len(u),vehicle_pixel_agreement=float(np.isin(im[v,u],[14,15,16,17,18,19]).mean()) if len(u) else None))
            if d in [-1,0] and i==60:
                panel=rgb.copy();draw=ImageDraw.Draw(panel)
                for xx,yy in zip(u[::3],v[::3]):draw.ellipse((xx-1,yy-1,xx+1,yy+1),fill=(255,60,40))
                panel=panel.resize((800,450));ImageDraw.Draw(panel).rectangle((0,0,800,30),fill='black');ImageDraw.Draw(panel).text((8,8),f'{route.split("_Town")[0]} image={i} OCC={i+d} (red vehicle voxels)',fill='white');panels.append(panel)
    canvas=Image.new('RGB',(1600,450*((len(panels)+1)//2)),(20,20,20))
    for j,p in enumerate(panels):canvas.paste(p,((j%2)*800,(j//2)*450))
    canvas.save(OUT/'rgb-occ-alignment.png')
    (OUT/'rgb-alignment.json').write_text(json.dumps({'offset_definition':'OCC image_frame+offset projected using camera-to-ego extrinsic from image frame','rows':rows,'limitations':'Filled boxes/historical voxels occlude each other; projected-voxel precision not pixel IoU, six front views/two routes only; not sensor timestamps'},indent=2))
    print(rows)
if __name__=='__main__':main()
