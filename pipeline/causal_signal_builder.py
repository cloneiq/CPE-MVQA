import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from models.causal_modules import CCSComputer, HCSSComputer
from pipeline.realtime_causal_effects import compute_ccs_realtime, compute_hcss_realtime
from pipeline.realtime_intervention_generator import get_interventions_for_sample_realtime


class StableDOSignalBuilder:
    """Frozen-teacher based signal generator for DO controller."""

    def __init__(
        self,
        ema_momentum: float = 0.9,
        eps: float = 1e-6,
        ccs_text_de_scale: float = 1.0,
        max_interventions: int = 10,
        sim_low: float = 0.47,
        relax_sim_low: float = 0.45,
        allow_last_resort_interventions: bool = True,
        store_fusion_bank: bool = True,
    ):
        self.ema_momentum = float(ema_momentum)
        self.eps = float(eps)
        self._ema: Dict[str, torch.Tensor] = {}
        self.hcss_computer = HCSSComputer()
        self.ccs_computer = CCSComputer(text_de_scale=float(ccs_text_de_scale))
        self.max_interventions = int(max_interventions)
        self.sim_low = float(sim_low)
        self.relax_sim_low = float(relax_sim_low)
        self.allow_last_resort_interventions = bool(allow_last_resort_interventions)
        self.store_fusion_bank = bool(store_fusion_bank)

    def _extract_logits(self, out):
        return out[0] if isinstance(out, tuple) else out

    def _entropy(self, p: torch.Tensor) -> torch.Tensor:
        h = -(p.clamp_min(self.eps) * torch.log(p.clamp_min(self.eps))).sum(dim=-1)
        return h / max(math.log(max(p.size(-1), 2)), self.eps)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean()
        std = x.std(unbiased=False).clamp_min(self.eps)
        z = (x - mu) / std
        return torch.tanh(z / 2.0)

    def _ema_smooth(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if name not in self._ema or self._ema[name].shape != x.shape:
            self._ema[name] = x.detach()
            return x
        smoothed = self.ema_momentum * self._ema[name] + (1.0 - self.ema_momentum) * x
        self._ema[name] = smoothed.detach()
        return smoothed

    @torch.no_grad()
    def _build_signals_connected(
        self,
        model_frozen,
        images: torch.Tensor,
        questions_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        intervention_bank,
        question_texts: Optional[List[str]],
        image_paths: Optional[List[str]],
        answer_types: Optional[List[str]],
        targets: Optional[torch.Tensor],
        common_kwargs: Optional[Dict],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if tokenizer is None:
            return None

        kwargs = dict(common_kwargs or {})
        kwargs["training"] = False
        kwargs["apply_do"] = False
        kwargs["causal_signals"] = None

        B = images.size(0)
        dev = images.device
        ccs_l, hcss_l, tde_l, vie_l, tie_l = [], [], [], [], []
        valid_l: List[float] = []
        error_l: List[str] = []
        fusion_fb_rows: List[Optional[List[float]]] = []

        for i in range(B):
            try:
                q_text = ""
                if question_texts is not None and i < len(question_texts):
                    q_text = (question_texts[i] or "").strip()
                if not q_text:
                    q_text = tokenizer.decode(questions_ids[i], skip_special_tokens=True)
                img_name = ""
                if image_paths is not None and i < len(image_paths):
                    img_name = str(image_paths[i] or "")
                ans_type = ""
                if answer_types is not None and i < len(answer_types):
                    ans_type = (answer_types[i] or "").strip().lower()
                ans_idx = int(targets[i].argmax().item()) if targets is not None else 0

                resp = get_interventions_for_sample_realtime(
                    img_name,
                    q_text,
                    intervention_bank,
                    answer_type=ans_type,
                    model=model_frozen,
                    tokenizer=tokenizer,
                    pure_encoder=None,
                    device="cpu",
                    max_interventions=self.max_interventions,
                    use_realtime_fallback=True,
                    min_quality_interventions=1,
                    sim_low=self.sim_low,
                    relax_sim_low=self.relax_sim_low,
                    allow_last_resort_interventions=self.allow_last_resort_interventions,
                    return_metadata=True,
                )
                interventions = resp.get("interventions", []) if isinstance(resp, dict) else []
                if not interventions:
                    ccs_l.append(0.0)
                    hcss_l.append(0.0)
                    tde_l.append(0.0)
                    vie_l.append(0.0)
                    tie_l.append(0.0)
                    valid_l.append(0.0)
                    fusion_fb_rows.append(None)
                    if isinstance(resp, dict):
                        error_l.append(str(resp.get("diag_empty_reason", "no_interventions")))
                    else:
                        error_l.append("no_interventions")
                    continue

                out_i = model_frozen(
                    images[i:i + 1],
                    questions_ids[i:i + 1],
                    attention_mask[i:i + 1],
                    **kwargs,
                )
                logits_i = self._extract_logits(out_i)
                fusion_row = None
                if self.store_fusion_bank:
                    fs = getattr(model_frozen, "_last_fusion_s", None)
                    if fs is not None and fs.dim() == 2 and fs.size(0) >= 1:
                        fusion_row = fs[0].detach().float().cpu().tolist()
                fusion_fb_rows.append(fusion_row)

                hcss_res = compute_hcss_realtime(
                    model_frozen,
                    images[i:i + 1],
                    q_text,
                    interventions,
                    tokenizer,
                    ans_idx,
                    logits_i,
                    self.hcss_computer,
                    answer_group=ans_type or None,
                    answer_idx_in_type=None,
                    device=str(dev),
                    min_interventions=1,
                    max_interventions=self.max_interventions,
                )
                ccs_res = compute_ccs_realtime(
                    model_frozen,
                    images[i:i + 1],
                    q_text,
                    interventions,
                    tokenizer,
                    ans_idx,
                    logits_i,
                    self.ccs_computer,
                    answer_group=ans_type or None,
                    answer_idx_in_type=None,
                    device=str(dev),
                    max_interventions=self.max_interventions,
                    precomputed_text_de=hcss_res.get("de_mean"),
                )
                ccs_l.append(float(ccs_res.get("ccs", 0.0)))
                hcss_l.append(float(hcss_res.get("hcss_scalar", 0.0)))
                tde_l.append(float(ccs_res.get("text_de", 0.0)))
                vie_l.append(float(ccs_res.get("visual_ie", 0.0)))
                tie_l.append(float(hcss_res.get("ie_mean", 0.0)))
                valid_l.append(1.0)
                error_l.append("")
            except Exception as e:
                ccs_l.append(0.0)
                hcss_l.append(0.0)
                tde_l.append(0.0)
                vie_l.append(0.0)
                tie_l.append(0.0)
                valid_l.append(0.0)
                fusion_fb_rows.append(None)
                error_l.append(str(e))

        return {
            "ccs_raw": torch.tensor(ccs_l, device=dev, dtype=torch.float32),
            "hcss_raw": torch.tensor(hcss_l, device=dev, dtype=torch.float32),
            "text_de_raw": torch.tensor(tde_l, device=dev, dtype=torch.float32),
            "vis_ie_raw": torch.tensor(vie_l, device=dev, dtype=torch.float32),
            "text_ie_raw": torch.tensor(tie_l, device=dev, dtype=torch.float32),
            "valid_mask": torch.tensor(valid_l, device=dev, dtype=torch.float32),
            "error_reason": error_l,
            "fusion_bank_rows": fusion_fb_rows,
        }

    @torch.no_grad()
    def build_signals(
        self,
        model_frozen,
        images: torch.Tensor,
        questions_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        do_questions_ids: Optional[torch.Tensor] = None,
        common_kwargs: Optional[Dict] = None,
        tokenizer=None,
        intervention_bank=None,
        question_texts: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        answer_types: Optional[List[str]] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        connected = self._build_signals_connected(
            model_frozen=model_frozen,
            images=images,
            questions_ids=questions_ids,
            attention_mask=attention_mask,
            tokenizer=tokenizer,
            intervention_bank=intervention_bank,
            question_texts=question_texts,
            image_paths=image_paths,
            answer_types=answer_types,
            targets=targets,
            common_kwargs=common_kwargs,
        )
        if connected is None:
            raise RuntimeError(
                "Signal builder requires tokenizer; intervention JSONL (--intervention_path) is optional "
                "(runtime + lexical fallbacks apply when missing)."
            )
        text_de = connected["text_de_raw"].clamp(0.0, 1.0)
        vis_ie = connected["vis_ie_raw"].clamp(0.0, 1.0)
        text_ie = connected["text_ie_raw"].clamp(0.0, 1.0)
        ccs_raw = connected["ccs_raw"]
        hcss = connected["hcss_raw"].clamp(0.0, 1.0)
        valid_mask = connected.get("valid_mask", torch.ones_like(ccs_raw))
        error_reason = connected.get("error_reason", [""] * int(ccs_raw.size(0)))

        text_de_n = self._normalize(text_de)
        vis_ie_n = self._normalize(vis_ie)
        ccs = self._normalize(ccs_raw)

        ccs = self._ema_smooth("ccs", ccs)
        ccs = torch.clamp(ccs, -2.0, 2.0)
        hcss = self._ema_smooth("hcss", hcss)
        text_de_n = self._ema_smooth("text_de", text_de_n)
        vis_ie_n = self._ema_smooth("vis_ie", vis_ie_n)
        text_ie = self._ema_smooth("text_ie", text_ie)
        ccs = ccs * 1.2
        text_de_n = text_de_n * 1.2
        vis_ie_n = vis_ie_n * 1.2

        return {
            "ccs": ccs.detach(),
            "hcss": hcss.detach(),
            "text_de": text_de_n.detach(),
            "vis_ie": vis_ie_n.detach(),
            "text_ie": text_ie.detach(),
            "valid_mask": valid_mask.detach(),
            "error_reason": error_reason,
            "fusion_bank_rows": connected.get("fusion_bank_rows"),
        }
