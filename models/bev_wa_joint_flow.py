"""Joint World-Action Flow Matching Architectures for Stage 4.

Implements:
1. UnifiedConditionEncoder: Encodes History BEVs, ego motion, route points, vehicle state, permissions.
2. Arm D (JointWorldActionPredictor): Simultaneous joint flow matching of future scene latents and trajectory.
   - Interactive attention: scene tokens attend to action tokens and vice versa.
   - Asymmetric gradient contract: stop-grad on L_scene -> action tokens; differentiable L_traj -> scene.
3. Arm B (ActionOnlyFlowPredictor): Flow matching of trajectory only from history condition.
4. Arm C (DecoupledFlowPredictor): Cascaded generation where predicted future scene latents condition trajectory flow.
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from configs.stage3 import Stage3Config
from models.latent import BEVEncoder
from models.flow_matcher import sinusoidal_time_embedding
from models.bev_wa_latent_flow import (
    TransformerDecoderLayerSDPA,
    BEVOnlineEncoder,
    EMATargetEncoder,
    MultiHorizonLatentFlowPredictor,
)


class UnifiedConditionEncoder(nn.Module):
    """Encodes all historical and planning context into a sequence of condition tokens.

    - History BEVs (4 frames x 64 tokens) + Ego Motion (4 tokens) = 260 tokens
    - Navigation route points (8 tokens)
    - Vehicle state & maneuver permissions (1 token)
    Total = 269 condition tokens of dimension D=256.
    """

    def __init__(self, cfg=None, encoder_ckpt=None, use_applied_action: bool = True):
        super().__init__()
        if cfg is None:
            cfg = Stage3Config()
        self.cfg = cfg
        self.bev_encoder = BEVOnlineEncoder(cfg, encoder_ckpt=encoder_ckpt)
        d = cfg.encoder_dim  # 256

        # Route points projection: [B, 8, 2] -> [B, 8, 256]
        self.route_proj = nn.Sequential(
            nn.Linear(2, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        # Vehicle state (4) + maneuver permissions (2) = 6 -> [B, 1, 256]
        self.vehicle_proj = nn.Sequential(
            nn.Linear(6, d),
            nn.GELU(),
            nn.Linear(d, d),
        )

        self.use_applied_action = use_applied_action
        if use_applied_action:
            # Applied action projection: [B, 4, 2] -> [B, 4, 256] (chassis steering & throttle feedback)
            self.action_proj = nn.Sequential(
                nn.Linear(2, d),
                nn.GELU(),
                nn.Linear(d, d),
            )
        else:
            self.action_proj = None

    def forward(
        self,
        history_bev,
        ego_motion,
        route_points,
        route_valid,
        vehicle,
        maneuver_allowed,
        applied_action=None,
        action_valid=None,
    ):
        b = history_bev.shape[0]

        # 1. Spatiotemporal BEV + ego motion: [B, 260, 256]
        bev_tokens = self.bev_encoder(history_bev, ego_motion)

        # 2. Route points: [B, 8, 256]
        route_feat = self.route_proj(route_points)
        if route_valid is not None:
            route_feat = route_feat * route_valid[..., None].float()

        # 3. Vehicle + permissions: [B, 1, 256]
        perm_float = maneuver_allowed.float() if maneuver_allowed.dtype == torch.bool else maneuver_allowed
        veh_perm = torch.cat([vehicle, perm_float], dim=-1)  # [B, 6]
        veh_token = self.vehicle_proj(veh_perm).unsqueeze(1)  # [B, 1, 256]

        tokens_to_cat = [bev_tokens, route_feat, veh_token]

        # 4. History applied action tokens (Protocol Section 1.3: [B, 4, 2] -> [B, 4, 256])
        if self.use_applied_action and self.action_proj is not None and applied_action is not None:
            action_tokens = self.action_proj(applied_action)  # [B, 4, 256]
            if action_valid is not None:
                action_tokens = action_tokens * action_valid[..., None].float()
            tokens_to_cat.append(action_tokens)

        # Combined condition tokens: [B, 269 or 273, 256]
        condition_tokens = torch.cat(tokens_to_cat, dim=1)
        return condition_tokens


class ArmDJointFlowModel(nn.Module):
    """Arm D: Joint World-Action Flow Matcher (New Mainline).

    Jointly generates future scene latents (256 tokens) and driving trajectory (10 waypoints)
    in a shared interactive Transformer backbone with asymmetric gradient flow.
    """

    def __init__(
        self,
        cfg=None,
        encoder_ckpt=None,
        d_model=384,
        nhead=6,
        nlayers=8,
        dim_feedforward=1536,
        scene_loss_weight=1.0,
        use_applied_action: bool = True,
    ):
        super().__init__()
        if cfg is None:
            cfg = Stage3Config()
        self.cfg = cfg
        self.d_model = d_model
        self.latent_dim = cfg.encoder_dim
        self.token_count = 64
        self.latent_dim = cfg.encoder_dim  # 256
        self.scene_loss_weight = scene_loss_weight

        # Encoders
        self.condition_encoder = UnifiedConditionEncoder(cfg, encoder_ckpt=encoder_ckpt, use_applied_action=use_applied_action)
        self.target_encoder = EMATargetEncoder(self.condition_encoder.bev_encoder)

        # Projections
        self.cond_proj = nn.Linear(self.latent_dim, d_model)
        self.scene_in_proj = nn.Linear(self.latent_dim, d_model)
        self.traj_in_proj = nn.Linear(4, d_model)

        # Embeddings
        self.scene_spatial_emb = nn.Parameter(torch.randn(1, 1, 64, d_model) * 0.02)
        self.scene_horizon_emb = nn.Parameter(torch.randn(1, 4, 1, d_model) * 0.02)
        self.traj_step_emb = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)
        self.modality_emb_scene = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.modality_emb_traj = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Continuous Flow Time Embedding
        self.time_emb_dim = 32
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Shared Interactive Transformer Decoder
        self.layers = nn.ModuleList([
            TransformerDecoderLayerSDPA(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ) for _ in range(nlayers)
        ])

        # Output velocity heads
        self.norm_scene = nn.LayerNorm(d_model)
        self.out_scene = nn.Linear(d_model, self.latent_dim)
        nn.init.xavier_uniform_(self.out_scene.weight, gain=0.02)
        nn.init.zeros_(self.out_scene.bias)

        self.norm_traj = nn.LayerNorm(d_model)
        self.out_traj = nn.Linear(d_model, 4)
        nn.init.xavier_uniform_(self.out_traj.weight, gain=0.02)
        nn.init.zeros_(self.out_traj.bias)

    def forward_interactive(self, z_scene, z_traj, s, condition_tokens=None, memory=None, detach_action_for_scene=True):
        """Passes joint scene and trajectory tokens through the interactive Transformer.

        Args:
            z_scene: [B, 4, 64, 256] noisy scene latents
            z_traj:  [B, 10, 4] noisy trajectory waypoints
            s:       [B] continuous flow time in [0, 1]
            condition_tokens: Optional [B, T_cond, 256]
            memory:  Optional precomputed projected condition [B, T_cond, d_model]
            detach_action_for_scene: bool (enforces asymmetric gradient contract)

        Returns:
            v_scene: [B, 4, 64, 256]
            v_traj:  [B, 10, 4]
        """
        b = z_scene.shape[0]

        # 1. Embed scene tokens: [B, 4, 64, d_model] -> [B, 256, d_model]
        scene_tokens = self.scene_in_proj(z_scene) + self.scene_spatial_emb + self.scene_horizon_emb + self.modality_emb_scene
        scene_tokens = scene_tokens.flatten(1, 2)

        # 2. Embed trajectory tokens: [B, 10, d_model]
        traj_tokens = self.traj_in_proj(z_traj) + self.traj_step_emb + self.modality_emb_traj

        # 3. Add flow time embedding
        time_emb = sinusoidal_time_embedding(s, self.time_emb_dim)  # [B, 32]
        time_feat = self.time_proj(time_emb)[:, None, :]  # [B, 1, d_model]
        scene_tokens = scene_tokens + time_feat
        traj_tokens = traj_tokens + time_feat

        joint_tokens = torch.cat([scene_tokens, traj_tokens], dim=1)

        # Project condition memory: [B, T_cond, d_model]
        if memory is None:
            memory = self.cond_proj(condition_tokens)

        # 4. Joint interactive Transformer pass with strict asymmetric gradient isolation
        for layer in self.layers:
            joint_tokens = layer(
                joint_tokens, memory,
                detach_action_for_scene=detach_action_for_scene,
                num_scene_tokens=256,
            )

        # 5. Split output tokens: first 256 scene tokens, last 10 trajectory tokens
        out_scene_tokens = joint_tokens[:, :256]
        out_traj_tokens = joint_tokens[:, 256:]

        v_scene = self.out_scene(self.norm_scene(out_scene_tokens)).reshape(b, 4, self.token_count, self.latent_dim)
        v_traj = self.out_traj(self.norm_traj(out_traj_tokens))

        return v_scene, v_traj

    def forward_train(self, batch):
        """Computes joint Optimal Transport Flow Matching loss with asymmetric gradient flow."""
        h = batch["history_bev"]
        ego = batch["ego_motion"]
        rp = batch["route_points"]
        rv = batch["route_valid"]
        veh = batch["vehicle"]
        ma = batch["maneuver_allowed"]
        fb = batch["future_bev"]
        fbm = batch["future_bev_mask"]
        traj = batch["trajectory"]
        trajm = batch["future_traj_mask"]
        action_mask = batch["action_mask"]
        applied_action = batch.get("applied_action")
        action_valid = batch.get("action_valid")

        b = h.shape[0]
        device = h.device

        # 1. Condition representation with optional history chassis applied action
        condition_tokens = self.condition_encoder(
            h, ego, rp, rv, veh, ma, applied_action=applied_action, action_valid=action_valid
        )

        # 2. Target scene latents from EMA teacher (eval, no_grad)
        with torch.no_grad():
            target_scene = self.target_encoder.encode_future(fb)  # [B, 4, 64, 256]

        target_traj = traj  # [B, 10, 4]

        # 3. Flow Matching noisy samples: s ~ U[0, 1]
        s = torch.rand(b, device=device, dtype=torch.float32)
        eps_scene = torch.randn_like(target_scene)
        eps_traj = torch.randn_like(target_traj)

        s_scene = s[:, None, None, None]
        s_traj = s[:, None, None]

        z_scene = (1.0 - s_scene) * eps_scene + s_scene * target_scene
        z_traj = (1.0 - s_traj) * eps_traj + s_traj * target_traj

        v_star_scene = target_scene - eps_scene
        v_star_traj = target_traj - eps_traj

        # 4. Interactive forward pass with asymmetric gradient isolation
        v_scene, v_traj = self.forward_interactive(
            z_scene, z_traj, s, condition_tokens, detach_action_for_scene=True
        )

        # 5. Trajectory Flow Loss (filtered by action qualification & valid steps)
        # trajm: [B, 10] bool; action_mask: [B] bool
        valid_traj_mask = (trajm.float() * action_mask[:, None].float())[..., None]  # [B, 10, 1]
        loss_traj_all = F.smooth_l1_loss(v_traj, v_star_traj, reduction="none") * valid_traj_mask
        loss_traj = loss_traj_all.sum() / torch.clamp(valid_traj_mask.sum() * 4, min=1.0)

        # 6. Scene Flow Loss (100% valid BEV transitions, weighted by scene_loss_weight)
        mask_scene = fbm.float()[:, :, None, None]  # [B, 4, 1, 1]
        loss_scene_all = (v_scene - v_star_scene).square() * mask_scene
        norm_scene = torch.clamp(mask_scene.sum() * (64 * self.latent_dim), min=1.0)
        loss_scene = (loss_scene_all.sum() / norm_scene) * self.scene_loss_weight

        total_loss = loss_traj + loss_scene

        return total_loss, {
            "loss_traj": float(loss_traj.item()),
            "loss_scene": float(loss_scene.item()),
            "total_loss": float(total_loss.item()),
        }

    @torch.no_grad()
    def solve_euler(
        self,
        history_bev,
        ego_motion,
        route_points,
        route_valid,
        vehicle,
        maneuver_allowed,
        applied_action=None,
        action_valid=None,
        num_steps=5,
        generator=None,
        noise_seed=None,
    ):
        """Simultaneously solves ODE for trajectory and future scene latents with fixed noise stream."""
        b = history_bev.shape[0]
        device = history_bev.device
        dtype = torch.float32

        condition_tokens = self.condition_encoder(
            history_bev,
            ego_motion,
            route_points,
            route_valid,
            vehicle,
            maneuver_allowed,
            applied_action=applied_action,
            action_valid=action_valid,
        )
        memory = self.cond_proj(condition_tokens)

        gen = generator
        if gen is None and noise_seed is not None:
            gen = torch.Generator(device=device).manual_seed(int(noise_seed))

        if gen is not None:
            z_scene = torch.randn(b, 4, self.token_count, self.latent_dim, device=device, dtype=dtype, generator=gen)
        else:
            z_scene = torch.randn(b, 4, self.token_count, self.latent_dim, device=device, dtype=dtype)

        # Deterministic zero initialization for Candidate 0 trajectory (per protocol section 2.3)
        z_traj = torch.zeros(b, 10, 4, device=device, dtype=dtype)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            s_val = i * dt
            s_batch = torch.full((b,), s_val, device=device, dtype=torch.float32)
            v_scene, v_traj = self.forward_interactive(
                z_scene, z_traj, s_batch, memory=memory, detach_action_for_scene=False
            )
            z_scene = z_scene + dt * v_scene
            z_traj = z_traj + dt * v_traj

        return z_traj, z_scene

    def update_teacher(self, tau=0.996):
        self.target_encoder.update_ema(self.condition_encoder.bev_encoder, tau=tau)


class ArmBActionOnlyFlowModel(nn.Module):
    """Arm B: Action-only Flow Matcher.

    Generates trajectory via flow matching conditioned only on history tokens.
    No future scene latents are modeled or interacted with.
    """

    def __init__(
        self,
        cfg=None,
        encoder_ckpt=None,
        d_model=384,
        nhead=6,
        nlayers=8,
        dim_feedforward=1536,
        use_applied_action: bool = True,
    ):
        super().__init__()
        if cfg is None:
            cfg = Stage3Config()
        self.cfg = cfg
        self.d_model = d_model
        self.latent_dim = cfg.encoder_dim

        # Encoders
        self.condition_encoder = UnifiedConditionEncoder(cfg, encoder_ckpt=encoder_ckpt, use_applied_action=use_applied_action)
        self.cond_proj = nn.Linear(cfg.encoder_dim, d_model)
        self.traj_in_proj = nn.Linear(4, d_model)
        self.traj_step_emb = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

        # Time embedding
        self.time_emb_dim = 32
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer Decoder
        self.layers = nn.ModuleList([
            TransformerDecoderLayerSDPA(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ) for _ in range(nlayers)
        ])

        self.norm_traj = nn.LayerNorm(d_model)
        self.out_traj = nn.Linear(d_model, 4)
        nn.init.xavier_uniform_(self.out_traj.weight, gain=0.02)
        nn.init.zeros_(self.out_traj.bias)

    def forward(self, z_traj, s, condition_tokens):
        b = z_traj.shape[0]
        traj_tokens = self.traj_in_proj(z_traj) + self.traj_step_emb
        time_feat = self.time_proj(sinusoidal_time_embedding(s, self.time_emb_dim))[:, None, :]
        traj_tokens = traj_tokens + time_feat

        memory = self.cond_proj(condition_tokens)
        for layer in self.layers:
            traj_tokens = layer(traj_tokens, memory)

        v_traj = self.out_traj(self.norm_traj(traj_tokens))
        return v_traj

    def forward_train(self, batch):
        h = batch["history_bev"]
        ego = batch["ego_motion"]
        rp = batch["route_points"]
        rv = batch["route_valid"]
        veh = batch["vehicle"]
        ma = batch["maneuver_allowed"]
        traj = batch["trajectory"]
        trajm = batch["future_traj_mask"]
        action_mask = batch["action_mask"]
        applied_action = batch.get("applied_action")
        action_valid = batch.get("action_valid")

        b = h.shape[0]
        device = h.device

        condition_tokens = self.condition_encoder(
            h, ego, rp, rv, veh, ma, applied_action=applied_action, action_valid=action_valid
        )
        target_traj = traj

        s = torch.rand(b, device=device, dtype=torch.float32)
        eps_traj = torch.randn_like(target_traj)
        s_traj = s[:, None, None]

        z_traj = (1.0 - s_traj) * eps_traj + s_traj * target_traj
        v_star_traj = target_traj - eps_traj

        v_traj = self.forward(z_traj, s, condition_tokens)

        valid_traj_mask = (trajm.float() * action_mask[:, None].float())[..., None]
        loss_traj_all = F.smooth_l1_loss(v_traj, v_star_traj, reduction="none") * valid_traj_mask
        loss_traj = loss_traj_all.sum() / torch.clamp(valid_traj_mask.sum() * 4, min=1.0)

        return loss_traj, {"loss_traj": float(loss_traj.item()), "total_loss": float(loss_traj.item())}

    @torch.no_grad()
    def solve_euler(
        self,
        history_bev,
        ego_motion,
        route_points,
        route_valid,
        vehicle,
        maneuver_allowed,
        applied_action=None,
        action_valid=None,
        num_steps=10,
        generator=None,
        noise_seed=None,
    ):
        b = history_bev.shape[0]
        device = history_bev.device
        condition_tokens = self.condition_encoder(
            history_bev,
            ego_motion,
            route_points,
            route_valid,
            vehicle,
            maneuver_allowed,
            applied_action=applied_action,
            action_valid=action_valid,
        )

        z_traj = torch.zeros(b, 10, 4, device=device, dtype=torch.float32)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            s_val = i * dt
            s_batch = torch.full((b,), s_val, device=device, dtype=torch.float32)
            v_traj = self.forward(z_traj, s_batch, condition_tokens)
            z_traj = z_traj + dt * v_traj

        return z_traj


class ArmCDecoupledFlowModel(nn.Module):
    """Arm C: Decoupled Cascaded Flow Matcher.

    Two-stage pipeline:
    1. Independent future latent predictor outputs predicted future scene latents.
    2. Future scene latents are concatenated to condition tokens to feed trajectory flow.
    """

    def __init__(
        self,
        cfg=None,
        encoder_ckpt=None,
        stage3_ckpt=None,
        d_model=384,
        nhead=6,
        nlayers=8,
        dim_feedforward=1536,
        use_applied_action: bool = True,
    ):
        super().__init__()
        if cfg is None:
            cfg = Stage3Config()
        self.cfg = cfg
        self.d_model = d_model
        self.latent_dim = cfg.encoder_dim

        # 1. Condition encoder
        self.condition_encoder = UnifiedConditionEncoder(cfg, encoder_ckpt=encoder_ckpt, use_applied_action=use_applied_action)

        # 2. Frozen Stage 3 Predictor for future latents
        self.stage3_predictor = MultiHorizonLatentFlowPredictor(
            token_count=64,
            latent_dim=cfg.encoder_dim,
            d_model=384,
            nhead=6,
            nlayers=8,
            dim_feedforward=1536,
        )
        if stage3_ckpt and Path(stage3_ckpt).exists():
            st3_state = torch.load(stage3_ckpt, map_location="cpu", weights_only=False)
            self.stage3_predictor.load_state_dict(st3_state["predictor"])
            print(f"Loaded frozen Stage 3 predictor from {stage3_ckpt}")
        for p in self.stage3_predictor.parameters():
            p.requires_grad = False
        self.stage3_predictor.eval()

        # 3. Trajectory Generator
        self.cond_proj = nn.Linear(cfg.encoder_dim, d_model)
        self.future_scene_proj = nn.Linear(cfg.encoder_dim, d_model)
        self.traj_in_proj = nn.Linear(4, d_model)
        self.traj_step_emb = nn.Parameter(torch.randn(1, 10, d_model) * 0.02)

        self.time_emb_dim = 32
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.layers = nn.ModuleList([
            TransformerDecoderLayerSDPA(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ) for _ in range(nlayers)
        ])

        self.norm_traj = nn.LayerNorm(d_model)
        self.out_traj = nn.Linear(d_model, 4)
        nn.init.xavier_uniform_(self.out_traj.weight, gain=0.02)
        nn.init.zeros_(self.out_traj.bias)

    def forward_train(self, batch):
        h = batch["history_bev"]
        ego = batch["ego_motion"]
        rp = batch["route_points"]
        rv = batch["route_valid"]
        veh = batch["vehicle"]
        ma = batch["maneuver_allowed"]
        traj = batch["trajectory"]
        trajm = batch["future_traj_mask"]
        action_mask = batch["action_mask"]
        applied_action = batch.get("applied_action")
        action_valid = batch.get("action_valid")

        b = h.shape[0]
        device = h.device

        # 1. Condition representation: [B, 269 or 273, 256]
        condition_tokens = self.condition_encoder(
            h, ego, rp, rv, veh, ma, applied_action=applied_action, action_valid=action_valid
        )

        # 2. Predicted future scene latents (detached from Stage 3 predictor)
        with torch.no_grad():
            cond_bev = condition_tokens[:, :260]
            pred_scene = self.stage3_predictor.solve_euler(cond_bev, num_steps=5).flatten(1, 2)  # [B, 256, 256]

        # 3. Combined condition: [B, T_cond + 256, d_model]
        mem_cond = self.cond_proj(condition_tokens)
        mem_future = self.future_scene_proj(pred_scene)
        memory = torch.cat([mem_cond, mem_future], dim=1)

        target_traj = traj
        s = torch.rand(b, device=device, dtype=torch.float32)
        eps_traj = torch.randn_like(target_traj)
        s_traj = s[:, None, None]

        z_traj = (1.0 - s_traj) * eps_traj + s_traj * target_traj
        v_star_traj = target_traj - eps_traj

        # Decoder pass
        traj_tokens = self.traj_in_proj(z_traj) + self.traj_step_emb
        time_feat = self.time_proj(sinusoidal_time_embedding(s, self.time_emb_dim))[:, None, :]
        traj_tokens = traj_tokens + time_feat

        for layer in self.layers:
            traj_tokens = layer(traj_tokens, memory)

        v_traj = self.out_traj(self.norm_traj(traj_tokens))

        valid_traj_mask = (trajm.float() * action_mask[:, None].float())[..., None]
        loss_traj_all = F.smooth_l1_loss(v_traj, v_star_traj, reduction="none") * valid_traj_mask
        loss_traj = loss_traj_all.sum() / torch.clamp(valid_traj_mask.sum() * 4, min=1.0)

        return loss_traj, {"loss_traj": float(loss_traj.item()), "total_loss": float(loss_traj.item())}

    @torch.no_grad()
    def solve_euler(
        self,
        history_bev,
        ego_motion,
        route_points,
        route_valid,
        vehicle,
        maneuver_allowed,
        applied_action=None,
        action_valid=None,
        num_steps=10,
        generator=None,
        noise_seed=None,
    ):
        b = history_bev.shape[0]
        device = history_bev.device

        condition_tokens = self.condition_encoder(
            history_bev,
            ego_motion,
            route_points,
            route_valid,
            vehicle,
            maneuver_allowed,
            applied_action=applied_action,
            action_valid=action_valid,
        )
        cond_bev = condition_tokens[:, :260]

        gen = generator
        if gen is None and noise_seed is not None:
            gen = torch.Generator(device=device).manual_seed(int(noise_seed))
        init_noise = (
            torch.randn(b, 4, 64, self.latent_dim, device=device, dtype=torch.float32, generator=gen)
            if gen is not None
            else None
        )
        pred_scene = self.stage3_predictor.solve_euler(cond_bev, num_steps=5, initial_noise=init_noise).flatten(1, 2)

        mem_cond = self.cond_proj(condition_tokens)
        mem_future = self.future_scene_proj(pred_scene)
        memory = torch.cat([mem_cond, mem_future], dim=1)

        z_traj = torch.zeros(b, 10, 4, device=device, dtype=torch.float32)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            s_val = i * dt
            s_batch = torch.full((b,), s_val, device=device, dtype=torch.float32)

            traj_tokens = self.traj_in_proj(z_traj) + self.traj_step_emb
            time_feat = self.time_proj(sinusoidal_time_embedding(s_batch, self.time_emb_dim))[:, None, :]
            traj_tokens = traj_tokens + time_feat

            for layer in self.layers:
                traj_tokens = layer(traj_tokens, memory)

            v_traj = self.out_traj(self.norm_traj(traj_tokens))
            z_traj = z_traj + dt * v_traj

        return z_traj
