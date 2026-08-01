from typing import Optional, Tuple, List, Dict
import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt


# --------------------------- 辅助函数 ---------------------------
def topk_mask_indices(scores: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """选择topk需要掩码的patch索引"""
    if scores.dim() == 1:
        lv = scores.shape[0]
        k = max(1, int(round(mask_ratio * lv)))
        if k >= lv:
            return torch.ones_like(scores, dtype=torch.bool)
        _, idx = torch.topk(scores, k)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[idx] = True
        return mask
    elif scores.dim() == 2:
        b, lv = scores.shape
        masks = torch.zeros_like(scores, dtype=torch.bool)
        k = max(1, int(round(mask_ratio * lv)))
        if k >= lv:
            masks[:] = True
            return masks
        topk = torch.topk(scores, k, dim=1).indices
        arange = torch.arange(b, device=scores.device).unsqueeze(1).expand(-1, k)
        masks[arange.reshape(-1), topk.reshape(-1)] = True
        return masks
    else:
        raise ValueError("scores must be 1d or 2d")


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """归一化分数到[0,1]"""
    if scores.dim() == 1:
        s = scores - scores.min()
        return s / (s.max() + 1e-12)
    else:
        s = scores - scores.min(dim=1, keepdim=True)[0]
        return s / (s.max(dim=1, keepdim=True)[0] + 1e-12)


# --------------------------- 掩码函数 ---------------------------
def mask_image_embeds(
    image_embeds: torch.Tensor,
    patch_scores: torch.Tensor,
    mask_ratio: float = 0.9,
    mask_mode: str = "zero",
    topk: Optional[int] = None
) -> torch.Tensor:
    """确保掩码与图像嵌入维度一致。topk: 固定 patch 数（覆盖 mask_ratio），用于 Local IE"""
    device = image_embeds.device
    b, lv, d = image_embeds.shape
    patch_scores = patch_scores.to(device)

    # topk 模式：固定 patch 数，更稳定（lesion 通常极小）
    if topk is not None:
        mask_ratio = min(topk, lv - 1) / max(lv, 1)

    # 适配 patch score 长度 (ViT通常有257个token, 其中一个CLS)
    # 假设输入 image_embeds 是 [B, L, D] (包含CLS或不包含，视模型而定)
    # patch_scores 应该是 [B, L] 或 [B, L-1]
    
    current_len = image_embeds.shape[1]
    score_len = patch_scores.shape[1]

    if score_len != current_len:
        # 常见情况：patch_scores不包含CLS，但image_embeds包含
        if score_len == current_len - 1:
             # 补一个0给CLS位置(假设CLS在0)
             patch_scores = F.pad(patch_scores, (1, 0), mode="constant", value=0)
        elif score_len > current_len:
            patch_scores = patch_scores[:, :current_len]
        else:
            pad_length = current_len - score_len
            patch_scores = F.pad(patch_scores, (0, pad_length), mode="constant", value=0)

    # 生成掩码
    norm_scores = normalize_scores(patch_scores)
    mask_bool = topk_mask_indices(norm_scores, mask_ratio)
    mask_bool = mask_bool.to(device)

    # 执行掩码
    masked = image_embeds.clone()
    if mask_mode == "zero":
        mask_expand = mask_bool.unsqueeze(-1).expand(-1, -1, d)
        masked[mask_expand] = 0.0
    elif mask_mode == "mean":
        mean_patch = image_embeds.mean(dim=1, keepdim=True)
        mask_expand = mask_bool.unsqueeze(-1).expand(-1, -1, d)
        masked = torch.where(mask_expand, mean_patch.expand(-1, current_len, -1), masked)
    elif mask_mode == "noise":
        noise_std = image_embeds.std().item() * 0.5
        noise = torch.randn_like(masked) * noise_std
        mask_expand = mask_bool.unsqueeze(-1).expand(-1, -1, d)
        masked[mask_expand] = noise[mask_expand]
    else:
        raise ValueError(f"unknown mask_mode: {mask_mode}")
    return masked


# --------------------------- 核心辅助：手动前向传播 ---------------------------
def manual_forward_pass_for_gradcam(model, images, question_input, device=None, return_logits=True):
    """
    手动执行 models/vqa_module.py 中 CausalVQAModel 的 forward 逻辑。
    为了 GradCAM，我们需要在 vision_encoder 输出后截获 tensor 并 track gradient。
    """
    if device is None:
        device = images.device

    # 1. 视觉编码
    # 注意：这里我们假设 images 已经是 tensor。
    # 为了 GradCAM，我们需要在这里允许梯度回传
    visual_embeds = model.vision_encoder(images)  # [B, L_v, D_v]
    
    # 关键点：如果是为了计算 inputs 的梯度，inputs本身要有 grad。
    # 但 GradCAM 通常计算的是 Feature Map 的梯度。
    # 我们在这里 detach 并 require_grad，以便计算关于 feature map 的梯度
    visual_embeds_leaf = visual_embeds.detach()
    visual_embeds_leaf.requires_grad = True
    
    # 继续前向传播
    uni_modal_image_feats = model.multi_modal_vision_proj(visual_embeds_leaf)
    
    # 2. 文本编码
    questions_ids = question_input["input_ids"]
    attention_mask = question_input["attention_mask"]
    
    uni_modal_text_feats = model.language_encoder.embeddings(input_ids=questions_ids)
    text_input_shape = attention_mask.size()
    extended_text_masks = model.language_encoder.get_extended_attention_mask(
        attention_mask, text_input_shape, questions_ids.device
    )
    
    for layer in model.language_encoder.encoder.layer:
        uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
    uni_modal_text_feats = model.multi_modal_language_proj(uni_modal_text_feats)

    # 3. 构造 Masks 和 Embeddings
    # 构造 image masks
    image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), 
                             dtype=torch.long, device=device)
    extended_image_masks = model.language_encoder.get_extended_attention_mask(
        image_masks, image_masks.size(), device
    )
    
    # 添加 modality embeddings
    # 注意：vqa_module.py 中使用 modality_type_embeddings(0) for text, (1) for image
    # 并且使用 image_token_type_idx=1
    image_token_type_idx = 1
    
    uni_modal_text_feats = uni_modal_text_feats + \
        model.modality_type_embeddings(torch.zeros_like(attention_mask))
    uni_modal_image_feats = uni_modal_image_feats + \
        model.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx))

    x_orig = uni_modal_text_feats
    x, y = uni_modal_text_feats, uni_modal_image_feats

    # 4. Two-Stage Co-Attention Loop (M3AE style)
    total_layers = len(model.multi_modal_language_layers)
    mid_point = total_layers // 2

    for i in range(mid_point):
        text_layer = model.multi_modal_language_layers[i]
        image_layer = model.multi_modal_vision_layers[i]
        x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
        y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
        x, y = x1[0], y1[0]

    # Reset text for stage 2
    x = x_orig
    for i in range(mid_point, total_layers):
        text_layer = model.multi_modal_language_layers[i]
        image_layer = model.multi_modal_vision_layers[i]
        x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
        y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
        x, y = x1[0], y1[0]

    # Optional post layers for backward compatibility
    if hasattr(model, "multi_modal_vision_post_layers") and len(model.multi_modal_vision_post_layers) > 0:
        for post_image_layer in model.multi_modal_vision_post_layers:
            y1 = post_image_layer(y, x, extended_image_masks, extended_text_masks)
            y = y1[0]

    # 6. Pooling
    multi_modal_text_cls_feats = model.multi_modal_language_pooler(x)
    multi_modal_image_cls_feats = model.multi_modal_vision_pooler(y)

    multi_modal_cls_feats = torch.cat(
        [multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)

    logits = model.vqa_head(multi_modal_cls_feats)
    
    return logits, visual_embeds_leaf


# --------------------------- 基于Grad-CAM的patch分数计算 ---------------------------
def get_gradcam_patch_score(
    model,
    image: torch.Tensor,
    question_input: Dict,
    answer_target: Optional[torch.Tensor] = None,
    device: torch.device = None,
    sample_data: Optional[Dict] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    计算 Grad-CAM 分数，适配新的 models/vqa_module.py 结构。
    """
    device = device or image.device
    meta = {"method": "gradcam", "success": False, "error": None}

    try:
        with torch.enable_grad():
            # 执行手动前向传播，捕获梯度
            logits, visual_embeds_leaf = manual_forward_pass_for_gradcam(model, image, question_input, device)

            # 确定目标类别
            if answer_target is None:
                target_ids = logits.argmax(dim=1)
            else:
                target_ids = answer_target.to(device).long()
                # 简单防越界
                target_ids = torch.clamp(target_ids, 0, logits.shape[1] - 1)

            # 获取目标 logits
            target_logits = logits.gather(1, target_ids.unsqueeze(1)).squeeze()
            
            # 反向传播，计算关于 visual_embeds_leaf 的梯度
            # sum() 处理 batch 情况
            model.zero_grad()
            target_logits.sum().backward()

            # 检查梯度
            grads = visual_embeds_leaf.grad
            if grads is None or torch.allclose(grads, torch.zeros_like(grads)):
                 # 有时候第一次 backward 可能有问题，或者计算图断了
                 raise RuntimeError("视觉嵌入梯度无效或全为零")
            
            # Grad-CAM 核心计算: Weights = Global Average Pooling of Gradients
            # visual_embeds_leaf shape: [B, L, D]
            # grads shape: [B, L, D]
            
            # 排除 CLS token (假设第一个是 CLS，通常 ViT/BERT 都是这样)
            # 如果是 Swin Transformer output 可能是 [B, L, D] 没有显式 CLS 这里需要确认
            # 大多数 ViT 实现输出 [B, N+1, D]。我们假设索引0是CLS。
            
            grads_content = grads[:, 1:, :]      # [B, N, D]
            acts_content = visual_embeds_leaf.data[:, 1:, :] # [B, N, D]
            
            # 计算权重 alpha
            alpha = grads_content.mean(dim=1) # [B, D] - Global Pool over patches
            
            # 加权求和: \sum alpha_k * A_k
            # broadcasting: [B, N, D] * [B, 1, D] -> sum dim=2 -> [B, N]
            gcam = (acts_content * alpha.unsqueeze(1)).sum(dim=2)
            
            # ReLU
            gcam = F.relu(gcam)
            
            meta.update({
                "success": True,
                "patch_num": gcam.shape[1],
                "grad_norm": torch.norm(grads).item()
            })
            return gcam, meta

    except Exception as e:
        meta["error"] = str(e)[:200]
        # 在出错时返回随机分数兜底，防止崩溃
        # 假设大约 576 个 patch (24x24)
        B = image.shape[0]
        return torch.rand(B, 576, device=device), meta


# --------------------------- 辅助函数：patch定位可视化 ---------------------------
def visualize_topk_patches(patch_scores, image_tensor, batch_idx, save_root, topk=30):
    """可视化关键patch"""
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.dim() == 3:
        if image_tensor.shape[0] in [1, 3]:
            image_tensor = image_tensor.unsqueeze(0)
        else:
            image_tensor = image_tensor.unsqueeze(1)
    elif image_tensor.dim() != 4:
        return

    B, C, H, W = image_tensor.shape
    if C == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)
        C = 3

    mean = torch.tensor([0.485, 0.456, 0.406]).to(image_tensor.device)[:C]
    std = torch.tensor([0.229, 0.224, 0.225]).to(image_tensor.device)[:C]
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    image_denorm = image_tensor * std + mean
    image_denorm = image_denorm.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    # 简单的 patch size 估算
    # 假设输入是 384x384, patch score 长度是 N
    # sqrt(N) * patch_size = 384
    num_patches = patch_scores.shape[1]
    grid_dim = int(np.sqrt(num_patches))
    if grid_dim * grid_dim != num_patches:
         # 非正方形grid，可能可视化不准
         pass
    
    patch_size = H // grid_dim

    for b in range(B):
        sample_scores = patch_scores[b].cpu().numpy()
        sample_image = image_denorm[b]
        
        if len(sample_scores) < topk:
            topk_indices = np.argsort(sample_scores)
        else:
            topk_indices = np.argsort(sample_scores)[-topk:]

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(sample_image)
        ax.axis('off')
        ax.set_title(f"Batch {batch_idx} | Sample {b} | Top {topk} Key Patches", fontsize=12, fontweight='bold')

        for idx in topk_indices:
            row = idx // grid_dim
            col = idx % grid_dim
            x1 = col * patch_size
            y1 = row * patch_size

            rect = plt.Rectangle((x1, y1), patch_size, patch_size,
                                  fill=False, color='red', linewidth=2, alpha=0.8)
            ax.add_patch(rect)

        save_path = os.path.join(save_root, f"batch_{batch_idx}_sample_{b}_key_patches.png")
        plt.tight_layout()
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        except:
            pass
        plt.close()


# --------------------------- 核心函数：生成掩码嵌入 ---------------------------
def generate_masked_embeddings(
    model,
    image,
    q_enc,
    mask_ratio=0.9,
    mask_mode="zero",
    heatmap_mode="gradcam",
    device="cuda",
    answer_targets=None,
    tokenizer=None,
    sample_data=None,
    visualize_patches=True,
    visualize_topk=30,
    visualize_batch_limit=10
):
    visualize_save_root = "visualizations/patch_visualize" # 修改为相对路径较安全
    if visualize_patches:
        os.makedirs(visualize_save_root, exist_ok=True)

    # 1. 准备输入
    # 确保 image 是 tensor
    image = image.to(device)
    B = image.shape[0]

    # 文本输入处理
    if isinstance(q_enc, list):
        # 如果是列表(Batch Encoding list)，拼接
        batch_input_ids = torch.cat([q["input_ids"] for q in q_enc], dim=0).to(device)
        batch_attention_mask = torch.cat([q["attention_mask"] for q in q_enc], dim=0).to(device)
    else:
        batch_input_ids = q_enc["input_ids"].to(device)
        batch_attention_mask = q_enc["attention_mask"].to(device)

    # 为了计算 GradCAM，需要启用梯度
    # 这里我们只开启必要的梯度，或者在 train 模式下运行
    model.train() # 必须 train 模式才能反向传播
    
    # 临时启用参数梯度
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad_(True)

    question_input = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask
    }

    # 2. 获取 Patch Scores (GradCAM)
    patch_scores, meta = get_gradcam_patch_score(
        model=model,
        image=image,
        question_input=question_input,
        answer_target=answer_targets,
        device=device,
        sample_data=sample_data
    )

    # 3. 可视化
    if visualize_patches and meta.get("success", False):
         # 使用 batch_idx 信息防止覆盖
         batch_idx = sample_data[0].get("meta_batch_idx", 0) if (sample_data and isinstance(sample_data, list)) else 0
         if batch_idx < visualize_batch_limit:
            visualize_topk_patches(
                patch_scores=patch_scores,
                image_tensor=image,
                batch_idx=batch_idx,
                save_root=visualize_save_root,
                topk=visualize_topk
            )

    # 4. 生成掩码后的 logits
    model.eval() # 推理模式
    with torch.no_grad():
        # 4.1 获取原始 Image Embeddings (在 proj 之前，即 Encoder 输出)
        # 用同样的 encoder 调用
        raw_image_embeds = model.vision_encoder(image) 
        # raw_image_embeds: [B, L, D] (含CLS)
        
        # 4.2 执行掩码
        # 注意：mask_image_embeds 会处理 CLS token 的对齐问题 (patch_scores 不带 CLS)
        masked_raw_embeds = mask_image_embeds(
            image_embeds=raw_image_embeds,
            patch_scores=patch_scores,
            mask_ratio=mask_ratio,
            mask_mode=mask_mode
        )
        
        # 定义后半部分的前向传播函数 (从 visual features 到 logits)
        def partial_forward_from_visual_embeds(visual_embs):
            # i. Projection
            uni_modal_image_feats = model.multi_modal_vision_proj(visual_embs)
            
            # ii. Text Encoding (Standard)
            uni_modal_text_feats = model.language_encoder.embeddings(input_ids=batch_input_ids)
            text_input_shape = batch_attention_mask.size()
            extended_text_masks = model.language_encoder.get_extended_attention_mask(
                batch_attention_mask, text_input_shape, batch_input_ids.device
            )
            for layer in model.language_encoder.encoder.layer:
                uni_modal_text_feats = layer(uni_modal_text_feats, extended_text_masks)[0]
            uni_modal_text_feats = model.multi_modal_language_proj(uni_modal_text_feats)
            
            # iii. Modality Embeddings & Masks
            image_masks = torch.ones((uni_modal_image_feats.size(0), uni_modal_image_feats.size(1)), 
                                     dtype=torch.long, device=device)
            extended_image_masks = model.language_encoder.get_extended_attention_mask(
                image_masks, image_masks.size(), device
            )
            
            image_token_type_idx = 1
            uni_modal_text_feats = uni_modal_text_feats + \
                model.modality_type_embeddings(torch.zeros_like(batch_attention_mask))
            uni_modal_image_feats = uni_modal_image_feats + \
                model.modality_type_embeddings(torch.full_like(image_masks, image_token_type_idx))
            
            x_orig = uni_modal_text_feats
            x, y = uni_modal_text_feats, uni_modal_image_feats

            total_layers = len(model.multi_modal_language_layers)
            mid_point = total_layers // 2
            for i in range(mid_point):
                text_layer = model.multi_modal_language_layers[i]
                image_layer = model.multi_modal_vision_layers[i]
                x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
                y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
                x, y = x1[0], y1[0]

            x = x_orig
            for i in range(mid_point, total_layers):
                text_layer = model.multi_modal_language_layers[i]
                image_layer = model.multi_modal_vision_layers[i]
                x1 = text_layer(x, y, extended_text_masks, extended_image_masks, output_attentions=True)
                y1 = image_layer(y, x, extended_image_masks, extended_text_masks, output_attentions=True)
                x, y = x1[0], y1[0]

            if hasattr(model, "multi_modal_vision_post_layers") and len(model.multi_modal_vision_post_layers) > 0:
                for post_image_layer in model.multi_modal_vision_post_layers:
                    y1 = post_image_layer(y, x, extended_image_masks, extended_text_masks)
                    y = y1[0]
                
            multi_modal_text_cls_feats = model.multi_modal_language_pooler(x)
            multi_modal_image_cls_feats = model.multi_modal_vision_pooler(y)
            multi_modal_cls_feats = torch.cat([multi_modal_text_cls_feats, multi_modal_image_cls_feats], dim=-1)
            return model.vqa_head(multi_modal_cls_feats)

        # 4.3 计算 Logits
        original_full_logits = partial_forward_from_visual_embeds(raw_image_embeds)
        masked_full_logits = partial_forward_from_visual_embeds(masked_raw_embeds)
        
        # 计算掩码效果统计
        mask_effect = (raw_image_embeds - masked_raw_embeds).abs().mean().item()

    # 恢复 param grad 状态
    for name, param in model.named_parameters():
        param.requires_grad_(original_requires_grad[name])

    return {
        "patch_scores": patch_scores,
        "original_full_logits": original_full_logits,
        "masked_full_logits": masked_full_logits,
        "meta": {
            "method": meta["method"],
            "mask_ratio": mask_ratio,
            "mask_mode": mask_mode,
            "mask_effect": mask_effect,
            "success": meta["success"],
            "error": meta.get("error", None)
        }
    }