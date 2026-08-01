"""
将旧 M3AE / baseline 单融合栈 checkpoint 键名映射到因果三栈模型。

必须在两处调用：
1) vqa_module 加载 load_path（m3ae.ckpt）时：仅映射融合层，不写入 language_encoder_cf（该模块此时尚未创建）。
2) main 加载 resume_from_baseline 时：映射融合层 + 可选复制 Q 塔权重到 CF 塔键名。
"""


def remap_m3ae_state_dict_to_causal_triple_stack(
    state_dict,
    *,
    copy_language_cf: bool = True,
    log_fn=None,
):
    """
    :param state_dict: ckpt 权重 dict（会复制，不原地改传入引用）
    :param copy_language_cf: False 用于 __init__ 中 load_path（encoder_cf 尚未注册）
    :param log_fn: 可选 callable(str)，例如 logger.info 或 print
    """
    sd = dict(state_dict)
    if any(k.startswith("stack_hcss.") for k in sd):
        return sd

    stacks = ("stack_hcss", "stack_cf", "stack_cem")
    additions = {}
    for k, v in sd.items():
        if k.startswith("multi_modal_language_layers."):
            rest = k[len("multi_modal_language_layers.") :]
            for st in stacks:
                additions[f"{st}.lang.{rest}"] = v
        elif k.startswith("multi_modal_vision_layers."):
            rest = k[len("multi_modal_vision_layers.") :]
            for st in stacks:
                additions[f"{st}.vis.{rest}"] = v
        elif k.startswith("multi_modal_language_pooler."):
            rest = k[len("multi_modal_language_pooler.") :]
            for st in stacks:
                additions[f"{st}.pool_t.{rest}"] = v
        elif k.startswith("multi_modal_vision_pooler."):
            rest = k[len("multi_modal_vision_pooler.") :]
            for st in stacks:
                additions[f"{st}.pool_v.{rest}"] = v
        elif k.startswith("fusion_layer_norm."):
            rest = k[len("fusion_layer_norm.") :]
            for st in stacks:
                additions[f"{st}.ln.{rest}"] = v

    if copy_language_cf:
        for k, v in sd.items():
            if k.startswith("language_encoder."):
                additions["language_encoder_cf." + k[len("language_encoder.") :]] = v
            elif k.startswith("multi_modal_language_proj.") and not k.startswith(
                "multi_modal_language_proj_cf"
            ):
                rest = k[len("multi_modal_language_proj.") :]
                additions[f"multi_modal_language_proj_cf.{rest}"] = v

    drop_prefixes = (
        "multi_modal_language_layers.",
        "multi_modal_vision_layers.",
        "multi_modal_language_pooler.",
        "multi_modal_vision_pooler.",
        "fusion_layer_norm.",
        "gate_layer_norm.",
    )
    out = {
        k: v
        for k, v in sd.items()
        if not any(k.startswith(p) for p in drop_prefixes)
    }
    out.update(additions)

    if additions and log_fn is not None:
        extra = " + CF text tower keys" if copy_language_cf else " (CF tower synced after load via deepcopy)"
        log_fn(
            f"  Remapped M3AE fusion -> 3×stack_hcss/cf/cem{extra}; "
            f"+{len(additions)} entries; dropped obsolete fusion key names."
        )
    return out
