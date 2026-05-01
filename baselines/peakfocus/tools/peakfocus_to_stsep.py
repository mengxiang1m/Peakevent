import numpy as np


def peakfocus_to_events(peak_probs, value_preds, threshold=0.4):
    """
    Convert PeakFocus dual-head output to STSEP event tuples for a single window.

    Args:
        peak_probs: [pred_len] sigmoid probabilities from peak head
        value_preds: [pred_len] de-normalized value predictions
        threshold: binary classification threshold

    Returns:
        List[Dict]: event dicts with onset/end/duration/apex/intensity.
    """
    peak_probs = np.asarray(peak_probs, dtype=np.float32).reshape(-1)
    value_preds = np.asarray(value_preds, dtype=np.float32).reshape(-1)

    if peak_probs.shape[0] != value_preds.shape[0]:
        raise ValueError(
            f'Length mismatch: len(peak_probs)={peak_probs.shape[0]} '
            f'!= len(value_preds)={value_preds.shape[0]}'
        )

    binary = (peak_probs >= threshold).astype(int)
    events = []
    i = 0

    while i < len(binary):
        if binary[i] == 1:
            block_start = i
            while i < len(binary) and binary[i] == 1:
                i += 1
            block_end = i

            duration = block_end - block_start
            apex_local = block_start + int(np.argmax(peak_probs[block_start:block_end]))
            apex_intensity = float(value_preds[apex_local])

            events.append(
                {
                    'onset_idx': int(block_start),
                    'end_idx': int(block_end),
                    'duration': int(duration),
                    'apex_idx': int(apex_local),
                    'apex_intensity': apex_intensity,
                }
            )
        else:
            i += 1

    return events
