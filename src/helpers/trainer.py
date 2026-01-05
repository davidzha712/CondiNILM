#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - PyTorch Trainer

#
#################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl


class EAECLoss(nn.Module):
    def __init__(
        self,
        threshold=10.0,
        alpha_on=3.0,
        alpha_off=1.0,
        lambda_grad=0.5,
        lambda_energy=0.5,
        soft_temp=10.0,
        edge_eps=5.0,
        energy_floor=1.0,
        lambda_sparse=0.0,
        lambda_zero=0.0,
        center_ratio=1.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.alpha_on = alpha_on
        self.alpha_off = alpha_off
        self.lambda_grad = lambda_grad
        self.lambda_energy = lambda_energy
        self.soft_temp = soft_temp
        self.edge_eps = edge_eps
        self.energy_floor = energy_floor
        self.lambda_sparse = lambda_sparse
        self.lambda_zero = lambda_zero
        self.center_ratio = float(center_ratio)
        self.base_loss = nn.SmoothL1Loss(reduction="none")
        self.grad_loss = nn.L1Loss(reduction="none")

    def forward(self, pred, target):
        pred = pred.float()
        target = target.float()
        time_len = pred.size(-1)
        ratio = float(self.center_ratio)
        if time_len > 1 and 0.0 < ratio < 1.0:
            center_len = max(1, int(round(time_len * ratio)))
            start = (time_len - center_len) // 2
            end = start + center_len
            pred = pred[..., start:end]
            target = target[..., start:end]
        loss_point = self.base_loss(pred, target)
        eps = 1e-6
        temp = max(self.soft_temp, eps)
        p_on = torch.sigmoid((target - self.threshold) / temp)
        p_off = 1.0 - p_on
        loss_on = (loss_point * p_on).sum() / (p_on.sum() + eps)
        loss_off = (loss_point * p_off).sum() / (p_off.sum() + eps)
        loss_main = self.alpha_on * loss_on + self.alpha_off * loss_off
        if pred.size(-1) > 1:
            pred_diff = pred[..., 1:] - pred[..., :-1]
            target_diff = target[..., 1:] - target[..., :-1]
            edge_mask = (target_diff.abs() > self.edge_eps).float()
            loss_grad_point = self.grad_loss(pred_diff, target_diff)
            loss_grad = (loss_grad_point * edge_mask).sum() / (
                edge_mask.sum() + eps
            )
        else:
            loss_grad = pred.new_tensor(0.0)
        energy_pred = pred.sum(dim=-1)
        energy_target = target.sum(dim=-1)
        floor = max(float(self.energy_floor), 1e-6)
        denom = energy_target.abs() + floor
        weight = energy_target.abs() / denom
        loss_energy_sample = ((energy_pred - energy_target).abs() / denom) * weight
        on_coverage = p_on.sum(dim=-1)
        min_on_steps = 0.1 * float(pred.size(-1))
        energy_mask = (on_coverage > min_on_steps).float()
        if energy_mask.sum() > 0:
            loss_energy = (loss_energy_sample * energy_mask).sum() / (
                energy_mask.sum() + eps
            )
        else:
            loss_energy = pred.new_tensor(0.0)
        sparse_penalty = (pred.abs() * p_on).sum() / (p_on.sum() + eps)
        zero_penalty = (pred.abs() * p_off).sum() / (p_off.sum() + eps)
        return (
            loss_main
            + self.lambda_grad * loss_grad
            + self.lambda_energy * loss_energy
            + self.lambda_sparse * sparse_penalty
            + self.lambda_zero * zero_penalty
        )


class GAEAECLoss(nn.Module):
    """
    Gate-Aware Energy-Aware Error Correction Loss.
    
    针对NILM问题优化，平衡处理：
    1. ON/OFF状态不平衡问题
    2. OFF状态时模型输出非零值的问题（假阳性）
    3. ON状态时模型输出太低的问题（漏检/假阴性）
    4. 显式门控学习，强化设备状态识别
    
    关键改进：添加ON漏检惩罚，防止模型学会"全部输出0"的安全策略
    """
    def __init__(
        self,
        threshold=10.0,
        alpha_on=3.0,
        alpha_off=1.0,
        lambda_grad=0.0,
        lambda_energy=0.5,
        soft_temp=10.0,
        edge_eps=5.0,
        energy_floor=1.0,
        lambda_sparse=0.0,
        lambda_zero=0.0,
        center_ratio=1.0,
        # OFF惩罚参数
        lambda_off_hard=0.1,    # OFF状态硬约束权重（降低！）
        off_margin=0.02,        # OFF状态的容忍边界（允许小噪声）
        # ON召回参数（新增！防止全0输出）
        lambda_on_recall=0.3,   # ON漏检惩罚权重
        on_recall_margin=0.5,   # ON时输出至少达到目标的多少比例
        # 门控分类参数
        lambda_gate_cls=0.1,    # 门控分类损失权重（降低！）
        gate_focal_gamma=2.0,   # Focal Loss gamma
    ):
        super().__init__()
        self.threshold = threshold
        self.alpha_on = alpha_on
        self.alpha_off = alpha_off
        self.lambda_energy = lambda_energy
        self.soft_temp = soft_temp
        self.energy_floor = energy_floor
        self.lambda_zero = lambda_zero
        self.lambda_off_hard = lambda_off_hard
        self.off_margin = off_margin
        self.lambda_on_recall = lambda_on_recall
        self.on_recall_margin = on_recall_margin
        self.lambda_gate_cls = lambda_gate_cls
        self.gate_focal_gamma = gate_focal_gamma
        self.center_ratio = float(center_ratio)
        self.base_loss = nn.SmoothL1Loss(reduction="none")

    def forward(self, pred, target, gate=None):
        pred = pred.float()
        target = target.float()
        time_len = pred.size(-1)
        ratio = float(self.center_ratio)
        if time_len > 1 and 0.0 < ratio < 1.0:
            center_len = max(1, int(round(time_len * ratio)))
            start = (time_len - center_len) // 2
            end = start + center_len
            pred = pred[..., start:end]
            target = target[..., start:end]
            if gate is not None:
                gate = gate[..., start:end]
        
        loss_point = self.base_loss(pred, target)
        eps = 1e-6
        thr = float(self.threshold)
        
        # 计算ON/OFF概率（基于目标值）
        temp = max(self.soft_temp, eps)
        p_on_target = torch.sigmoid((target - thr) / temp)
        p_off_target = 1.0 - p_on_target
        
        # 硬掩码（基于真实目标）
        hard_off_mask = (target <= thr).float()
        hard_on_mask = (target > thr).float()
        
        # ==================== 主损失：加权回归损失 ====================
        # 使用目标值的软权重，而不是门控输出
        loss_on = (loss_point * p_on_target).sum() / (p_on_target.sum() + eps)
        loss_off = (loss_point * p_off_target).sum() / (p_off_target.sum() + eps)
        loss_main = self.alpha_on * loss_on + self.alpha_off * loss_off
        
        # ==================== 能量损失 ====================
        energy_pred_on = (pred * p_on_target).sum(dim=-1)
        energy_target_on = (target * p_on_target).sum(dim=-1)
        floor = max(float(self.energy_floor), 1e-6)
        denom = energy_target_on.abs() + floor
        loss_energy_sample = (energy_pred_on - energy_target_on).abs() / denom
        on_coverage = p_on_target.sum(dim=-1)
        min_on_steps = 0.1 * float(pred.size(-1))
        energy_mask = (on_coverage > min_on_steps).float()
        if energy_mask.sum() > 0:
            loss_energy = (loss_energy_sample * energy_mask).sum() / (energy_mask.sum() + eps)
        else:
            loss_energy = pred.new_tensor(0.0)
        
        # ==================== OFF假阳性惩罚（温和版本） ====================
        # 只惩罚OFF时超过margin的输出，使用线性惩罚而非平方
        if hard_off_mask.sum() > 0 and self.lambda_off_hard > 0:
            margin = max(float(self.off_margin), 0.0)
            pred_excess = torch.relu(pred.abs() - margin)
            # 使用线性惩罚（更温和）
            off_fp_penalty = (pred_excess * hard_off_mask).sum() / (hard_off_mask.sum() + eps)
        else:
            off_fp_penalty = pred.new_tensor(0.0)
        
        # ==================== ON漏检惩罚（关键！防止全0输出） ====================
        # 惩罚ON时输出太低的情况
        if hard_on_mask.sum() > 0 and self.lambda_on_recall > 0:
            # 计算ON时预测值与目标值的差距
            recall_margin = max(float(self.on_recall_margin), 0.1)
            # 目标：pred >= target * recall_margin
            min_expected = target * recall_margin
            # 惩罚低于预期的输出
            on_shortfall = torch.relu(min_expected - pred)
            on_fn_penalty = (on_shortfall * hard_on_mask).sum() / (hard_on_mask.sum() + eps)
        else:
            on_fn_penalty = pred.new_tensor(0.0)
        
        # ==================== 门控分类损失（平衡版本） ====================
        if gate is not None and self.lambda_gate_cls > 0:
            gate_target = hard_on_mask
            gate_prob = torch.sigmoid(gate.float())
            gate_prob = torch.clamp(gate_prob, eps, 1.0 - eps)
            
            # 计算ON/OFF比例
            on_ratio = hard_on_mask.mean()
            off_ratio = 1.0 - on_ratio
            
            # 平衡的类别权重：少数类获得更高权重
            # 但限制权重比例，避免极端不平衡
            weight_on = torch.clamp(off_ratio / (on_ratio + eps), 1.0, 5.0)
            weight_off = 1.0
            
            # Binary Cross Entropy with class weights
            bce_on = -torch.log(gate_prob) * gate_target * weight_on
            bce_off = -torch.log(1.0 - gate_prob) * (1.0 - gate_target) * weight_off
            
            # Focal modulation（可选，gamma=0时退化为普通BCE）
            gamma = max(float(self.gate_focal_gamma), 0.0)
            if gamma > 0:
                pt = gate_prob * gate_target + (1.0 - gate_prob) * (1.0 - gate_target)
                focal_weight = (1.0 - pt) ** gamma
                gate_cls_loss = (focal_weight * (bce_on + bce_off)).mean()
            else:
                gate_cls_loss = (bce_on + bce_off).mean()
        else:
            gate_cls_loss = pred.new_tensor(0.0)
        
        # ==================== 软零惩罚（原有，保持温和） ====================
        if hard_off_mask.sum() > 0 and self.lambda_zero > 0:
            zero_penalty = (pred.abs() * hard_off_mask).sum() / (hard_off_mask.sum() + eps)
        else:
            zero_penalty = pred.new_tensor(0.0)
        
        # ==================== 组合总损失 ====================
        total_loss = (
            loss_main 
            + self.lambda_energy * loss_energy 
            + self.lambda_zero * zero_penalty
            + self.lambda_off_hard * off_fp_penalty      # OFF假阳性惩罚
            + self.lambda_on_recall * on_fn_penalty      # ON漏检惩罚（新增！）
            + self.lambda_gate_cls * gate_cls_loss
        )
        
        return total_loss


class SeqToSeqLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-2,
        criterion=None,
        patience_rlr=None,
        n_warmup_epochs=0,
        warmup_type="linear",
        neg_penalty_weight=0.1,
        rlr_factor=0.1,
        rlr_min_lr=0.0,
        state_zero_penalty_weight=0.0,
        zero_run_kernel=0,
        zero_run_ratio=0.8,
        loss_threshold=0.0,
        off_high_agg_penalty_weight=0.0,
        gate_cls_weight=0.0,
        gate_window_weight=0.0,
        gate_focal_gamma=2.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        if criterion is None:
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = criterion
        self.patience_rlr = patience_rlr
        self.n_warmup_epochs = n_warmup_epochs
        self.warmup_type = warmup_type
        self.neg_penalty_weight = float(neg_penalty_weight)
        self.rlr_factor = float(rlr_factor)
        self.rlr_min_lr = float(rlr_min_lr)
        self.state_zero_penalty_weight = float(state_zero_penalty_weight)
        self.zero_run_kernel = int(zero_run_kernel)
        self.zero_run_ratio = float(zero_run_ratio)
        self.loss_threshold = float(loss_threshold)
        self.off_high_agg_penalty_weight = float(off_high_agg_penalty_weight)
        self.gate_cls_weight = float(gate_cls_weight)
        self.gate_window_weight = float(gate_window_weight)
        self.gate_focal_gamma = float(gate_focal_gamma)
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def forward(self, ts_agg):
        return self.model(ts_agg)

    def _gate_focal_bce(self, logits, targets):
        if self.gate_cls_weight <= 0.0:
            return logits.new_tensor(0.0)
        logits = logits.float()
        targets = targets.float()
        probs = torch.sigmoid(logits)
        eps = 1e-6
        probs = torch.clamp(probs, eps, 1.0 - eps)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_pos = 1.5
        alpha_neg = 1.0
        alpha = alpha_pos * targets + alpha_neg * (1.0 - targets)
        gamma = max(self.gate_focal_gamma, 0.0)
        loss = -alpha * ((1.0 - pt) ** gamma) * torch.log(pt)
        return loss.mean()

    def _gate_window_bce(self, logits, state):
        if self.gate_window_weight <= 0.0:
            return logits.new_tensor(0.0)
        logits = logits.float()
        state = state.float()
        window_label = (state.sum(dim=-1, keepdim=True) > 0.5).float()
        pooled = logits.mean(dim=-1, keepdim=True)
        pooled = pooled.view(pooled.size(0), -1)
        window_label = window_label.view(window_label.size(0), -1)
        return F.binary_cross_entropy_with_logits(pooled, window_label)

    def _zero_run_penalty(self, pred, target):
        if self.state_zero_penalty_weight <= 0.0 or self.zero_run_kernel <= 1:
            return pred.new_tensor(0.0)
        eps = 1e-6
        thr = max(float(self.loss_threshold), 0.0)
        target = target.float()
        target_abs = target.abs()
        if thr > 0.0:
            zero_mask = (target_abs <= thr).float()
        else:
            zero_mask = (target_abs < eps).float()
        min_len = min(int(self.zero_run_kernel), target.size(-1))
        if min_len <= 1:
            return pred.new_tensor(0.0)
        long_zero_mask = torch.zeros_like(zero_mask)
        B, C, L = zero_mask.shape
        for b in range(B):
            for c in range(C):
                row = zero_mask[b, c]  # [L]
                if row.sum() <= 0:
                    continue
                diff = row[1:] - row[:-1]
                starts = (diff == 1).nonzero(as_tuple=False).flatten() + 1
                ends = (diff == -1).nonzero(as_tuple=False).flatten() + 1
                if row[0] > 0.5:
                    starts = torch.cat(
                        [torch.tensor([0], device=row.device, dtype=starts.dtype), starts]
                    )
                if row[-1] > 0.5:
                    ends = torch.cat(
                        [ends, torch.tensor([L], device=row.device, dtype=ends.dtype)]
                    )
                if starts.numel() == 0 or ends.numel() == 0:
                    continue
                n_seg = min(starts.numel(), ends.numel())
                starts = starts[:n_seg]
                ends = ends[:n_seg]
                for s, e in zip(starts.tolist(), ends.tolist()):
                    if e - s >= min_len:
                        long_zero_mask[b, c, s:e] = 1.0
        if long_zero_mask.sum() <= 0:
            return pred.new_tensor(0.0)
        p_abs = pred.float().abs()
        if thr > 0.0:
            p_above = torch.relu(p_abs - thr)
        else:
            p_above = p_abs
        penalty = (p_above * long_zero_mask).sum() / (long_zero_mask.sum() + eps)
        return self.state_zero_penalty_weight * penalty

    def _off_high_agg_penalty(self, pred, target, ts_agg):
        if self.off_high_agg_penalty_weight <= 0.0:
            return pred.new_tensor(0.0)
        eps = 1e-6
        thr_t = max(float(self.loss_threshold), 0.0)
        target = target.float()
        target_abs = target.abs()
        if thr_t > 0.0:
            off_mask = (target_abs <= thr_t).float()
        else:
            off_mask = (target_abs < eps).float()
        ts_agg = ts_agg.float().abs()
        if ts_agg.ndim == 3:
            agg_main = ts_agg[:, 0, :]
        else:
            agg_main = ts_agg
        max_per = agg_main.amax(dim=-1, keepdim=True)
        high_thr = 0.5 * max_per
        high_mask = (agg_main >= high_thr).float()
        if off_mask.ndim == 3 and high_mask.ndim == 2:
            high_mask = high_mask.unsqueeze(1)
        joint_mask = off_mask * high_mask
        if joint_mask.sum() <= 0:
            return pred.new_tensor(0.0)
        p_abs = pred.float().abs()
        if thr_t > 0.0:
            excess = torch.relu(p_abs - thr_t)
        else:
            excess = p_abs
        penalty = (excess * joint_mask).sum() / (joint_mask.sum() + eps)
        return self.off_high_agg_penalty_weight * penalty

    def training_step(self, batch, batch_idx):
        ts_agg, appl, state = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(self.criterion, GAEAECLoss) and hasattr(
            self.model, "forward_with_gate"
        ):
            pred, gate = self.model.forward_with_gate(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target, gate=gate)
            gate_cls_loss = self._gate_focal_bce(gate, state)
            gate_window_loss = self._gate_window_bce(gate, state)
        else:
            pred = self(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target)
            gate_cls_loss = pred.new_tensor(0.0)
            gate_window_loss = pred.new_tensor(0.0)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        neg_penalty = torch.relu(-pred).mean()
        neg_penalty = torch.nan_to_num(neg_penalty, nan=0.0, posinf=1e4, neginf=-1e4)
        zero_run_penalty = self._zero_run_penalty(pred, target)
        zero_run_penalty = torch.nan_to_num(
            zero_run_penalty, nan=0.0, posinf=1e4, neginf=-1e4
        )
        off_high_agg_penalty = self._off_high_agg_penalty(pred, target, ts_agg)
        off_high_agg_penalty = torch.nan_to_num(
            off_high_agg_penalty, nan=0.0, posinf=1e4, neginf=-1e4
        )
        loss = (
            loss_main
            + zero_run_penalty
            + off_high_agg_penalty
            + self.neg_penalty_weight * neg_penalty
            + self.gate_cls_weight * gate_cls_loss
            + self.gate_window_weight * gate_window_loss
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_agg, appl, _ = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if isinstance(self.criterion, GAEAECLoss) and hasattr(
            self.model, "forward_with_gate"
        ):
            pred, gate = self.model.forward_with_gate(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            gate = torch.nan_to_num(gate, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target, gate=gate)
        else:
            pred = self(ts_agg)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_main = self.criterion(pred, target)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        neg_penalty = torch.relu(-pred).mean()
        neg_penalty = torch.nan_to_num(neg_penalty, nan=0.0, posinf=1e4, neginf=-1e4)
        zero_run_penalty = self._zero_run_penalty(pred, target)
        zero_run_penalty = torch.nan_to_num(
            zero_run_penalty, nan=0.0, posinf=1e4, neginf=-1e4
        )
        off_high_agg_penalty = self._off_high_agg_penalty(pred, target, ts_agg)
        off_high_agg_penalty = torch.nan_to_num(
            off_high_agg_penalty, nan=0.0, posinf=1e4, neginf=-1e4
        )
        loss = (
            loss_main
            + zero_run_penalty
            + off_high_agg_penalty
            + self.neg_penalty_weight * neg_penalty
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedulers = []
        if self.n_warmup_epochs and self.n_warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch >= self.n_warmup_epochs:
                    return 1.0
                if self.warmup_type == "linear":
                    return float(epoch + 1) / float(self.n_warmup_epochs)
                if self.warmup_type == "exponential":
                    return float(0.5 ** (self.n_warmup_epochs - epoch - 1))
                if self.warmup_type == "constant":
                    return 0.1
                raise ValueError(f"Unsupported warmup type: {self.warmup_type}")

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
            schedulers.append(
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                }
            )

        if self.patience_rlr is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.patience_rlr,
                factor=self.rlr_factor,
                min_lr=self.rlr_min_lr,
                eps=1e-7,
            )
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                }
            )

        if not schedulers:
            return optimizer

        return [optimizer], schedulers

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if (
            self.current_epoch >= self.n_warmup_epochs
            and val_loss <= self.best_val_loss
        ):
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }


class TserLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=1e-2,
        criterion=None,
        patience_rlr=None,
        n_warmup_epochs=0,
        warmup_type="linear",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        self.patience_rlr = patience_rlr
        self.n_warmup_epochs = n_warmup_epochs
        self.warmup_type = warmup_type
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def forward(self, ts_agg):
        return self.model(ts_agg)

    def training_step(self, batch, batch_idx):
        ts_agg, target = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        pred = self(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_agg, target = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        pred = self(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedulers = []
        if self.n_warmup_epochs and self.n_warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch >= self.n_warmup_epochs:
                    return 1.0
                if self.warmup_type == "linear":
                    return float(epoch + 1) / float(self.n_warmup_epochs)
                if self.warmup_type == "exponential":
                    return float(0.5 ** (self.n_warmup_epochs - epoch - 1))
                if self.warmup_type == "constant":
                    return 0.1
                raise ValueError(f"Unsupported warmup type: {self.warmup_type}")

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
            schedulers.append(
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                }
            )

        if self.patience_rlr is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.patience_rlr,
                eps=1e-7,
            )
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                }
            )

        if not schedulers:
            return optimizer

        return [optimizer], schedulers

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if (
            self.current_epoch >= self.n_warmup_epochs
            and val_loss <= self.best_val_loss
        ):
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }


class DiffNILMLightningModule(pl.LightningModule):
    def __init__(self, model, criterion=None):
        super().__init__()
        self.model = model
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def forward(self, ts_agg):
        self.model.eval()
        return self.model(ts_agg)

    def training_step(self, batch, batch_idx):
        seqs, labels_energy, status = batch
        seqs = torch.nan_to_num(seqs.float(), nan=0.0, posinf=0.0, neginf=0.0)
        labels_energy = torch.nan_to_num(
            labels_energy.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        status = torch.nan_to_num(status.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.train()
        loss = self.model((seqs, labels_energy, status))
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_agg, appl, _ = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.eval()
        pred = self.model(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        if hasattr(self.model, "optimizer"):
            return self.model.optimizer
        return optim.Adam(self.model.parameters())

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if val_loss <= self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }


class STNILMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience_rlr=None,
        n_warmup_epochs=0,
        warmup_type="linear",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if hasattr(model, "criterion") and isinstance(model.criterion, nn.Module):
            self.criterion = model.criterion
        else:
            self.criterion = nn.MSELoss()
        self.weight_moe = getattr(model, "weight_moe", 0.0)
        self.patience_rlr = patience_rlr
        self.n_warmup_epochs = n_warmup_epochs
        self.warmup_type = warmup_type
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.best_model_state_dict = None
        self.loss_train_history = []
        self.loss_valid_history = []

    def forward(self, ts_agg):
        self.model.eval()
        return self.model(ts_agg)

    def training_step(self, batch, batch_idx):
        seqs, labels, status = batch
        seqs = torch.nan_to_num(seqs.float(), nan=0.0, posinf=0.0, neginf=0.0)
        labels = torch.nan_to_num(labels.float(), nan=0.0, posinf=0.0, neginf=0.0)
        status = torch.nan_to_num(status.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.train()
        power_logits, loss_moe = self.model(seqs)
        power_logits = torch.nan_to_num(
            power_logits, nan=0.0, posinf=1e4, neginf=-1e4
        )
        loss_moe = torch.nan_to_num(loss_moe, nan=0.0, posinf=1e4, neginf=-1e4)
        loss_main = self.criterion(power_logits, labels)
        loss_main = torch.nan_to_num(loss_main, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = loss_main + self.weight_moe * loss_moe
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ts_agg, appl, _ = batch
        ts_agg = torch.nan_to_num(ts_agg.float(), nan=0.0, posinf=0.0, neginf=0.0)
        target = torch.nan_to_num(appl.float(), nan=0.0, posinf=0.0, neginf=0.0)
        self.model.eval()
        pred = self.model(ts_agg)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        loss = self.criterion(pred, target)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        schedulers = []
        if self.n_warmup_epochs and self.n_warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch >= self.n_warmup_epochs:
                    return 1.0
                if self.warmup_type == "linear":
                    return float(epoch + 1) / float(self.n_warmup_epochs)
                if self.warmup_type == "exponential":
                    return float(0.5 ** (self.n_warmup_epochs - epoch - 1))
                if self.warmup_type == "constant":
                    return 0.1
                raise ValueError(f"Unsupported warmup type: {self.warmup_type}")

            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
            schedulers.append(
                {
                    "scheduler": warmup_scheduler,
                    "interval": "epoch",
                }
            )

        if self.patience_rlr is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                patience=self.patience_rlr,
                eps=1e-7,
            )
            schedulers.append(
                {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                }
            )

        if not schedulers:
            return optimizer

        return [optimizer], schedulers

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "train_loss" in metrics:
            self.loss_train_history.append(float(metrics["train_loss"]))

    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        if "val_loss" in metrics:
            val_loss = float(metrics["val_loss"])
            self.loss_valid_history.append(val_loss)
        writer = None
        if self.trainer is not None and self.trainer.logger is not None:
            if hasattr(self.trainer.logger, "experiment"):
                writer = self.trainer.logger.experiment
        if writer is not None:
            epoch_idx = int(self.current_epoch)
            for log_key, log_val in self.trainer.callback_metrics.items():
                if isinstance(log_val, (int, float)):
                    writer.add_scalar(log_key, float(log_val), epoch_idx)
        if (
            self.current_epoch >= self.n_warmup_epochs
            and val_loss <= self.best_val_loss
        ):
            self.best_val_loss = val_loss
            self.best_epoch = int(self.current_epoch)
            self.best_model_state_dict = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }
