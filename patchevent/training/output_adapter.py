"""Model output normalization helpers."""

from __future__ import annotations

def _unpack_model_outputs(model_out):
    logits = model_out
    int_preds = None
    int_cls_logits = None
    int_positions = None
    future_values_norm = None
    events_batch = None
    event_onset = None
    event_dur = None
    event_apex = None
    event_valid = None
    predicted_intensity = None
    count_logits = None
    pos_reg_preds = None
    if isinstance(model_out, dict):
        logits = model_out['logits']
        int_preds = model_out.get('int_preds')
        int_cls_logits = model_out.get('int_cls_logits')
        int_positions = model_out.get('int_positions')
        future_values_norm = model_out.get('future_values_norm')
        events_batch = model_out.get('events_batch')
        event_onset = model_out.get('event_onset')
        event_dur = model_out.get('event_dur')
        event_apex = model_out.get('event_apex')
        event_valid = model_out.get('event_valid')
        predicted_intensity = model_out.get('predicted_intensity')
        count_logits = model_out.get('count_logits')
        pos_reg_preds = model_out.get('pos_reg_preds')
    elif isinstance(model_out, tuple):
        if len(model_out) == 2:
            logits, int_preds = model_out
        elif len(model_out) == 3:
            logits, int_cls_logits, int_positions = model_out
        else:
            logits = model_out[0]
    return logits, int_preds, int_cls_logits, int_positions, future_values_norm, events_batch, event_onset, event_dur, event_apex, event_valid, predicted_intensity, count_logits, pos_reg_preds


# Backward compatibility alias
unpack_model_outputs = _unpack_model_outputs


__all__ = ["unpack_model_outputs", "_unpack_model_outputs"]
