"""Structured event tokenizer definitions."""

from __future__ import annotations

import json as _json

import torch

class StructuredEventTokenizer:
    """
    结构化事件tokenizer，词汇表大小随pred_len动态调整。
    每个事件 = 4 tokens: [ONSET, DUR, APEX, INT]
    完整序列: BOS [ONSET DUR APEX INT]* EOS

    VOCAB_SIZE = 3 + pred_len + 3 + pred_len + 100
      pred_len=96  → 298 (默认，向后兼容)
      pred_len=168 → 442
      pred_len=336 → 778
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    N_DUR = 3
    N_INT = 100
    INT_MIN = 0.50
    INT_STEP = 0.02        # 100 bins: [0.50, 2.48]

    def __init__(self, pred_len: int = 96, event_schema: str = 'quad'):
        self.pred_len = pred_len
        self.event_schema = event_schema
        self.has_intensity = event_schema == 'quad'
        if event_schema not in ('quad', 'triplet'):
            raise ValueError(f'Unsupported event_schema: {event_schema}')
        self.event_size = 4 if self.has_intensity else 3
        self.intensity_pos = 3 if self.has_intensity else None
        self.N_ONSET = pred_len
        self.N_APEX = pred_len
        self.ONSET_OFFSET = 3                          # ONSET_k = 3 + k,   k ∈ [0, pred_len-1]
        self.DUR_OFFSET = 3 + pred_len                 # DUR_d = 3+pred_len + (d-3), d ∈ {3,4,5}
        self.APEX_OFFSET = 3 + pred_len + 3            # APEX_k = 3+pred_len+3 + k
        self.INT_OFFSET = 3 + pred_len + 3 + pred_len
        self.VOCAB_SIZE = self.INT_OFFSET + (self.N_INT if self.has_intensity else 0)

    def encode(self, json_str: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        """JSON字符串 → token ids"""
        data = _json.loads(json_str)
        events = data.get('peak_events', [])
        ids: list[int] = []
        if add_bos:
            ids.append(self.BOS_ID)
        for ev in events:
            onset = max(0, min(self.N_ONSET - 1, int(ev[0])))
            ids.append(self.ONSET_OFFSET + onset)
            dur = max(3, min(5, int(ev[1])))
            ids.append(self.DUR_OFFSET + (dur - 3))
            apex = max(0, min(self.N_APEX - 1, int(ev[2])))
            ids.append(self.APEX_OFFSET + apex)
            if self.has_intensity:
                intensity = float(ev[3])
                bin_idx = int(round((intensity - self.INT_MIN) / self.INT_STEP))
                bin_idx = max(0, min(self.N_INT - 1, bin_idx))
                ids.append(self.INT_OFFSET + bin_idx)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode_events(self, ids: list[int]) -> list[dict]:
        events: list[dict] = []
        i = 0
        while i < len(ids) and ids[i] in (self.PAD_ID, self.BOS_ID):
            i += 1
        while i < len(ids):
            if ids[i] == self.EOS_ID or ids[i] == self.PAD_ID:
                break
            need = self.event_size - 1
            if i + need >= len(ids):
                break
            onset_id, dur_id, apex_id = ids[i], ids[i+1], ids[i+2]
            onset = onset_id - self.ONSET_OFFSET
            if not (0 <= onset < self.N_ONSET):
                break
            dur = (dur_id - self.DUR_OFFSET) + 3
            if not (3 <= dur <= 5):
                break
            apex = apex_id - self.APEX_OFFSET
            if not (0 <= apex < self.N_APEX):
                break
            intensity = float('nan')
            if self.has_intensity:
                int_id = ids[i + 3]
                int_bin = int_id - self.INT_OFFSET
                if not (0 <= int_bin < self.N_INT):
                    break
                intensity = round(self.INT_MIN + int_bin * self.INT_STEP, 2)
            events.append({
                'onset': onset,
                'duration': dur,
                'apex_index': apex,
                'apex_intensity': intensity,
            })
            i += self.event_size
        return events

    def decode(self, ids: list[int]) -> str:
        """token ids → JSON字符串"""
        events = []
        for ev in self.decode_events(ids):
            item = [ev['onset'], ev['duration'], ev['apex_index']]
            if self.has_intensity:
                item.append(ev['apex_intensity'])
            events.append(item)
        return _json.dumps({"peak_events": events}, separators=(',', ':'))

    def get_attr_positions(self, total_len: int, attr: str, device: torch.device) -> torch.Tensor:
        attr_to_idx = {'onset': 0, 'duration': 1, 'apex': 2}
        if self.has_intensity:
            attr_to_idx['intensity'] = 3
        if attr not in attr_to_idx:
            raise ValueError(f'Unsupported attr: {attr}')
        positions = torch.arange(attr_to_idx[attr], total_len, self.event_size, device=device)
        return positions[positions < total_len]

    def get_position_valid_mask(self, seq_len: int) -> torch.Tensor:
        """位置约束掩码: (seq_len, VOCAB_SIZE) bool"""
        mask = torch.zeros(seq_len, self.VOCAB_SIZE, dtype=torch.bool)
        for k in range(seq_len):
            r = k % self.event_size
            if r == 0:
                mask[k, self.ONSET_OFFSET:self.ONSET_OFFSET + self.N_ONSET] = True
                mask[k, self.EOS_ID] = True
            elif r == 1:
                mask[k, self.DUR_OFFSET:self.DUR_OFFSET + self.N_DUR] = True
            elif r == 2:
                mask[k, self.APEX_OFFSET:self.APEX_OFFSET + self.N_APEX] = True
            elif self.has_intensity:
                mask[k, self.INT_OFFSET:self.INT_OFFSET + self.N_INT] = True
        return mask


# Backward compatibility alias
EventTokenizer = StructuredEventTokenizer


__all__ = ["StructuredEventTokenizer", "EventTokenizer"]
