"""Authorized post-ViT spatial pooling; original pixels and ViT are unchanged."""
import math
import torch
from torch.nn import functional as F
from qwen_drive.scene import CAMERA_VIEWS,REASONING_REQUEST


def budget_grids(grids,text_count,limit=850):
    # Native merger already reduces patch grids by2. This is an EXTRA2x2 pool.
    if any(int(t)!=1 or int(h)%2 or int(w)%2 for t,h,w in grids):
        raise ValueError('Expected individual images with even native patch grids')
    native=[(int(t),int(h)//2,int(w)//2) for t,h,w in grids]
    pooled=[(t,math.ceil(h/2),math.ceil(w/2)) for t,h,w in native]
    budget=limit-text_count
    if budget<len(pooled):raise ValueError('Prompt leaves no visual token budget')
    target=list(pooled)
    # Retain every current camera; additional spatial reduction is explicit.
    while sum(t*h*w for t,h,w in target)>budget:
        i=max(range(len(target)),key=lambda j:target[j][1]*target[j][2])
        t,h,w=target[i]
        if h>=w and h>1:h-=1
        elif w>1:w-=1
        else:raise ValueError('Cannot satisfy token budget')
        target[i]=(t,h,w)
    return native,pooled,target


def planning_inputs(processor,scene,device='cuda',limit=850):
    # Use the original patchifier and each frame's original target_size.
    frames=[]; patches=[];grids=[];spec=[]
    views=list(CAMERA_VIEWS)+[view for view in scene.views if view not in CAMERA_VIEWS]
    for view in views:
        source=scene.views[view]
        if not source:raise ValueError(f'Camera {view} has no current frame')
        indices=range(len(source)) if view==CAMERA_VIEWS[0] else [len(source)-1]
        for i in indices:
            frame=source[i]
            pixels=processor.config.current_image_pixels if i==len(source)-1 else processor.config.history_image_pixels
            p,(h,w)=processor._patchify(frame,pixels)
            patches.append(p);grids.append((1,h,w))
            current=i==len(source)-1
            spec.append((view,scene.num_camera_frames-1 if current else i,current))
            frames.append({'view':view,'frame_index':i,'current':i==len(source)-1,'patch_grid':[h,w],
                           'processor_pixel_size':[w*processor.config.image_patch_size,h*processor.config.image_patch_size]})
    instruction=scene.instruction()
    start=instruction.find('1. Historical trajectories')
    if start<0:raise ValueError('Cannot preserve native ego/navigation text without the history section')
    instruction=('Current frame: every supplied camera view. Historical RGB: front camera only; original frame indices retained.\n'
                 +instruction[start:]+REASONING_REQUEST)
    def prompt(counts):
        body=[];previous=None
        for (view,index,current),count in zip(spec,counts):
            if previous!=view:body+=processor._encode(view);previous=view
            body+=processor._encode(f'frame: {index}')
            body += [processor.vision_start_id]+[processor.image_token_id]*count+[processor.vision_end_id]
        body+=processor._encode(instruction)
        return ([processor.im_start_id]+processor._encode('user')+processor.newline_ids+body+
                [processor.im_end_id]+processor.newline_ids+[processor.im_start_id]+
                processor._encode('assistant')+processor.newline_ids)
    text_count=len(prompt([0]*len(grids)))
    native,pooled,target=budget_grids(grids,text_count,limit)
    ids=prompt([t*h*w for t,h,w in target])
    assert len(ids)<=limit
    import numpy as np
    inputs={'input_ids':torch.tensor([ids],device=device),
            'pixel_values':torch.cat(patches).to(device),
            'image_grid_thw':torch.tensor(grids,device=device),
            'llm_grid_thw':torch.tensor([(t,h*2,w*2) for t,h,w in target],device=device),
            'history':torch.as_tensor(np.asarray(scene.history),device=device)[None],
            'history_velocity':torch.as_tensor(scene.history_velocity,device=device)[None],
            'history_acceleration':torch.as_tensor(scene.history_acceleration,device=device)[None],
            'ego_status':torch.as_tensor(scene.ego_status,device=device)[None],
            'nav_command':torch.tensor([scene.nav_command],device=device)}
    inputs['input_audit']={'frames':frames,'native_grids':native,'pooled_2x2_grids':pooled,'final_grids':target,
                          'text_tokens':text_count,'total_tokens':len(ids),'raw_resolution_changed':False,
                          'dropped_history_side_images':sum(len(v) for v in scene.views.values())-len(frames)}
    return inputs


def repeat_scene(inputs,batch_size):
    # Verification fixture only: materialize all pixels for each batch member.
    out={k:v for k,v in inputs.items()}
    for k,v in inputs.items():
        if isinstance(v,torch.Tensor):out[k]=v.repeat(batch_size,*([1]*(v.ndim-1)))
    return out


def compressed_forward(model,inputs):
    vlm=model.vlm.model
    with torch.no_grad():
        visual=vlm.visual(inputs['pixel_values'],grid_thw=inputs['image_grid_thw']).pooler_output
        counts=[int(t*h*w)//4 for t,h,w in inputs['image_grid_thw']]
        pieces=visual.split(counts)
        images=[]
        for values,(_,h,w),(_,lh,lw) in zip(pieces,inputs['image_grid_thw'],inputs['llm_grid_thw']):
            h,w,lh,lw=int(h)//2,int(w)//2,int(lh)//2,int(lw)//2
            grid=values.reshape(h,w,-1).permute(2,0,1)[None]
            pooled=F.avg_pool2d(grid,2,ceil_mode=True,count_include_pad=False)
            if pooled.shape[-2:]!=(lh,lw):pooled=F.adaptive_avg_pool2d(pooled,(lh,lw))
            images.append(pooled[0].permute(1,2,0).reshape(-1,values.shape[-1]))
        embeds=vlm.get_input_embeddings()(inputs['input_ids'])
        image_values=torch.cat(images).to(embeds.dtype)
        mask=(inputs['input_ids']==model.config.vlm_config.image_token_id).unsqueeze(-1).expand_as(embeds)
        assert int(mask.sum())==image_values.numel()
        embeds=embeds.masked_scatter(mask,image_values)
        positions=model._rope_positions(inputs['input_ids'],inputs['llm_grid_thw'])
    return vlm.language_model(inputs_embeds=embeds,position_ids=positions,use_cache=False)
