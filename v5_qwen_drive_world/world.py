"""Action-conditioned corridor extension of the released perception stack.

The native FPN, depth network, view transform, BEV transformer and occupancy
decoder are reused. Geometry and initialization changes are explicit below.
This is a new untrained extension, not a pretrained 160m perception capability.
"""
import copy
import os
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .perception_ops import rocm_perception_ops, native_fp32_batch_norm, align_deformable_output_dtype
from .runtime import math_attention


@contextmanager
def world_operator_context():
    with math_attention(), rocm_perception_ops(collapse_camera_sum=True):
        yield


class ActionConditionedWorld(nn.Module):
    def __init__(self, perception, image_size=(896, 512), grid=None):
        super().__init__()
        grid = grid or {"x_min": -16., "x_max": 112., "y_min": -19.2, "y_max": 19.2,
                        "height": 128, "width": 128, "dx": 1., "dy": .3}
        self.grid = grid; self.height = grid["height"]; self.width = grid["width"]
        self.native = perception.bev_modeling
        self.native.float()
        # Native NCDHW convs with complete depth halos bound im2col workspace.
        # NDHWC passed one shape but stalled in full encoder backward selection.
        os.environ['PYTORCH_MIOPEN_SUGGEST_NHWC'] = '0'
        from .conv3d import use_depth_tiled_conv3d
        self.tiled_conv3d_layers = use_depth_tiled_conv3d(self.native)
        native_fp32_batch_norm(self.native)
        # Recompute native 3D residual blocks independently so the decoder's
        # backward does not retain all full-resolution block activations at once.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl
        decoder = self.native.head.transformer.occ_decoder
        self.decoder_checkpoint_blocks = 0
        for stage in decoder.children():
            if hasattr(stage, 'blocks'):
                for index, block in enumerate(stage.blocks):
                    stage.blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
                    self.decoder_checkpoint_blocks += 1
        for module in self.native.modules():
            if module.__class__.__name__ == "MSDeformableAttention3D":
                module.register_forward_hook(align_deformable_output_dtype, with_kwargs=True)
        self.pc_range = [grid["x_min"], grid["y_min"], -5., grid["x_max"], grid["y_max"], 5.4]
        head = self.native.head; transformer = head.transformer
        old_range = list(head.pc_range); device = head.bev_embedding.weight.device
        old_h, old_w = head.bev_h, head.bev_w
        y = torch.linspace(grid["y_min"] + grid["dy"]/2, grid["y_max"] - grid["dy"]/2, self.height, device=device)
        x = torch.linspace(grid["x_min"] + grid["dx"]/2, grid["x_max"] - grid["dx"]/2, self.width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        normalized = torch.stack((2*(xx-old_range[0])/(old_range[3]-old_range[0])-1,
                                  2*(yy-old_range[1])/(old_range[4]-old_range[1])-1), -1)
        # Physical reprojection of learned query priors. Outside old support is
        # zero initialization, NOT free-space labels or inferred observations.
        query = head.bev_embedding.weight.detach().reshape(old_h, old_w, -1).permute(2,0,1)[None]
        query = F.grid_sample(query, normalized[None], align_corners=False, padding_mode="zeros")
        head.bev_embedding = nn.Embedding.from_pretrained(query[0].permute(1,2,0).reshape(-1,256), freeze=False)
        head.bev_h = self.height; head.bev_w = self.width
        head.pc_range = self.pc_range
        transformer.det_pc_range = self.pc_range
        transformer.encoder.pc_range = self.pc_range
        for layer in transformer.encoder.layers:
            for attention in layer.attentions:
                if hasattr(attention, "pc_range"): attention.pc_range = self.pc_range
        # Positional embeddings are trainable normalized-grid priors, not pixels.
        for name, size in (("row_embed", self.height), ("col_embed", self.width)):
            embedding = getattr(head.positional_encoding, name).weight.detach()
            resized = F.interpolate(embedding.T[None], size=size, mode="linear", align_corners=False)[0].T
            setattr(head.positional_encoding, name, nn.Embedding.from_pretrained(resized, freeze=False))
        view = self.native.view_trans
        view.pc_range = self.pc_range
        view.voxel_shape = [self.width, self.height, 16]
        view.voxel_size = [grid["dx"], grid["dy"], (self.pc_range[5]-self.pc_range[2])/16]
        # Keep existing pretrained 1..60m depth bins, explicitly initialize the
        # new bins out to160m; copied near-range bins are not stretched in space.
        view.frustum_range = [0,0,1., image_size[0], image_size[1],160.]
        view.frustum_size = [16.,16.,.5]; view.depth_dim = 318; view._frustum = None
        previous = self.native.depth_net.depth_conv[-1]
        extended = nn.Conv2d(previous.in_channels, 318, 1, device=device, dtype=torch.float32)
        with torch.no_grad():
            extended.weight[:previous.out_channels].copy_(previous.weight)
            extended.bias[:previous.out_channels].copy_(previous.bias)
            extended.weight[previous.out_channels:].zero_()
            extended.bias[previous.out_channels:].fill_(-8.)
        self.native.depth_net.depth_conv[-1] = extended
        self.native.depth_net.depth_channels = 318
        # No extra input encoder. These are only the requested future residual head.
        self.action_condition = nn.Sequential(nn.Linear(102,256), nn.SiLU(), nn.Linear(256,512)).to(device)
        self.delta = nn.Sequential(nn.Conv2d(256,256,3,padding=1), nn.SiLU(), nn.Conv2d(256,256,1)).to(device)
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
        self.migration = {"old_bev": [old_h,old_w], "new_bev": [self.height,self.width],
                          "query_transfer": "physical grid_sample; unsupported initialization zero",
                          "depth_bins": {"copied": previous.out_channels, "new":318-previous.out_channels},
                          "perception_weights_reused": True, "detection_and_map_heads": "preserved but not used by future dynamic World",
                          "learned_far_range_capability": False}
        # Native layer checkpointing also bounds the encoder recomputation peak.
        for index, layer in enumerate(view.conv_layer):
            view.conv_layer[index] = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
        for index, layer in enumerate(transformer.encoder.layers):
            transformer.encoder.layers[index] = checkpoint_wrapper(
                layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                context_fn=lambda: (world_operator_context(),world_operator_context()))

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained BatchNorm statistics: checkpoint recomputation must
        # not update them twice. Affine parameters remain trainable in SFT.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    def encode_current(self, vit, language, img_metas):
        """Native current scene features; process scenes separately for calibration.

        Official released UVTR averages batches in its helper. Call with B=1
        per scene until a separately verified batched geometry path is available.
        vit:[N,H,W,1024], language:[N,H/2,W/2,2560].
        """
        if vit.ndim != 4 or language.ndim != 4 or len(img_metas) != 1:
            raise ValueError("Per-scene official perception requires B=1 and calibrated cameras")
        n = language.shape[0]
        with rocm_perception_ops(collapse_camera_sum=True):
            features = self.native.adaptor(language.permute(0,3,1,2))
            features = [f.reshape(1,n,*f.shape[1:]) for f in features]
            vision_features = self.native.vit_neck(vit.permute(0,3,1,2))
            depth = [self.native.depth_net(f).softmax(1) for f in vision_features]
            volume = self.native.view_trans([f.reshape(1,n,*f.shape[1:]) for f in vision_features],
                                           img_depth=depth, img_metas=copy.deepcopy(img_metas))
            uvtr = self.native._uvtr_voxel_to_bev_tokens(volume)
            head = self.native.head
            queries = (head.bev_embedding.weight + uvtr).to(features[0].dtype)
            pos = head.positional_encoding(queries.new_zeros((1,self.height,self.width))).to(features[0].dtype)
            latent = head.transformer.get_bev_features(features, queries, self.height, self.width,
                                                       bev_pos=pos, img_metas=img_metas)
        latent = latent.transpose(1,2).reshape(1,16,16,self.height,self.width)
        fused = head.transformer._fuse_uvtr_occ_feat(latent, volume)
        return fused.reshape(1,256,self.height,self.width)

    def _decode(self, latent):
        transformer = self.native.head.transformer
        volume = latent.reshape(len(latent),16,16,self.height,self.width)
        decoded = transformer.occ_decoder(volume).permute(0,4,3,2,1)
        logits = transformer.occ_pred_head(decoded).float()
        dynamic = logits.softmax(-1)[..., [0,2,4]].sum(-1).amax(-1)
        # Native voxel output is X,Y,Z; external BEV output is Y,X.
        return dynamic.transpose(-1,-2).clamp(1e-6,1-1e-6)

    def forward(self, current, trajectory_xy, v0):
        if trajectory_xy.shape[-2:] != (50,2):
            raise ValueError("World conditions on the native50-step5s physical trajectory")
        results = []; deltas = []
        for horizon in (1.,2.,3.):
            action = torch.cat((trajectory_xy.flatten(1)/160., v0[:,None]/35.,
                                v0.new_full((len(v0),1),horizon/3.)),1)
            shift, scale = self.action_condition(action).chunk(2,dim=-1)
            residual = self.delta(current*(1+scale[:,:,None,None])+shift[:,:,None,None])
            future = current.detach()+residual
            probability = checkpoint(self._decode,future,use_reentrant=False) if self.training else self._decode(future)
            results.append(probability); deltas.append(residual)
        return {"occupancy": torch.stack(results,1), "delta_latent": torch.stack(deltas,1)}
