"""Extract all v2 ablation results for README."""
import json, numpy as np, os

def read_test_summary(path):
    p = os.path.join(path, 'test_summary.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)

def collect_dir(base, cfgs, seeds=(42, 123, 456)):
    results = {}
    for cfg in cfgs:
        vals = {}
        for s in seeds:
            d = os.path.join(base, f"{cfg}_s{s}")
            m = read_test_summary(d)
            if m:
                vals[str(s)] = m
        if vals:
            results[cfg] = {'per_seed': vals, 'n_done': len(vals)}
    return results

# Collect new encoder results directly from checkpoints
NEW_ENC_CFGS = ['GS4_lstm', 'GS4_mlp', 'GS4_scratch', 'GS5_detr']
for domain in ['wlel', 'ett', 'elc']:
    base = f'patchevent/phase2/checkpoints/{domain}_arch_ablation_v2'
    new_enc = collect_dir(base, NEW_ENC_CFGS)
    print(f"\n=== NEW ENCODERS: {domain.upper()} ===")
    print(f"{'Config':<22} {'F1':>12} {'OnMAE':>12} {'ApMAE':>12} {'DurMAE':>10}")
    print("-"*75)
    for cfg, cdata in new_enc.items():
        f1s, on, ap, dur = [], [], [], []
        for s, m in cdata['per_seed'].items():
            f1s.append(m.get('test_event_f1', m.get('event_f1', 0)))
            on.append(m.get('test_onset_mae', m.get('onset_mae', 0)))
            ap.append(m.get('test_apex_mae', m.get('apex_mae', 0)))
            dur.append(m.get('test_duration_mae', m.get('duration_mae', 0)))
        if f1s:
            print(f"{cfg:<22} {np.mean(f1s):>8.3f}±{np.std(f1s):.3f}  {np.mean(on):>8.3f}±{np.std(on):.3f}  {np.mean(ap):>8.3f}±{np.std(ap):.3f}  {np.mean(dur):>8.3f}±{np.std(dur):.3f}")


data = json.load(open('patchevent/phase2/checkpoints/ablation_v2_summary.json'))

def summarize(cfg_data):
    seeds_data = cfg_data.get('per_seed', {})
    f1s, on, ap, dur, mape = [], [], [], [], []
    for s, m in seeds_data.items():
        f1s.append(m.get('event_f1', 0))
        on.append(m.get('onset_mae', 0))
        ap.append(m.get('apex_mae', 0))
        dur.append(m.get('duration_mae', 0))
        mape.append(m.get('intensity_mape', 0))
    if not f1s:
        return None
    return {
        'f1': (np.mean(f1s), np.std(f1s)),
        'on': (np.mean(on), np.std(on)),
        'ap': (np.mean(ap), np.std(ap)),
        'dur': (np.mean(dur), np.std(dur)),
        'mape': (np.mean(mape), np.std(mape)),
        'n': len(f1s)
    }

def fmt(m, s): return f"{m:.3f}±{s:.3f}"
def fmtm(m, s): return f"{m:.4f}±{s:.4f}"

print("\n" + "="*100)
print("GS Architecture Ablation v2")
print("="*100)

arch = data['arch_v2']
order = ['G0_full', 'GS1_wo_mempos', 'GS2_wo_sa', 'GS3_bare', 'GS4_cnn',
         'GS4_lstm', 'GS4_mlp', 'GS4_scratch', 'GS5_detr']

for domain in ['wlel', 'ett', 'elc']:
    print(f"\n--- {domain.upper()} ---")
    print(f"{'Config':<22} {'F1':>12} {'OnMAE':>12} {'ApMAE':>12} {'DurMAE':>10} {'IntMAPE':>14}")
    print("-"*85)
    dom_data = arch.get(domain, {})
    for cfg in order:
        if cfg not in dom_data:
            print(f"{cfg:<22} {'--':>12}")
            continue
        s = summarize(dom_data[cfg])
        if s is None:
            print(f"{cfg:<22} {'--':>12}")
            continue
        print(f"{cfg:<22} {fmt(*s['f1']):>12} {fmt(*s['on']):>12} {fmt(*s['ap']):>12} {fmt(*s['dur']):>10} {fmtm(*s['mape']):>14}")

print("\n" + "="*100)
print("GD AR Ablation v2")
print("="*100)

ar = data['ar_v2']
ar_order = ['GD1_no_pos_mask', 'GD2_no_pos_smooth', 'GD3_no_attr_weight', 'GD4_no_causal']

for domain in ['wlel', 'ett', 'elc']:
    print(f"\n--- {domain.upper()} ---")
    print(f"{'Config':<25} {'F1':>12} {'OnMAE':>12} {'ApMAE':>12} {'DurMAE':>10}")
    print("-"*75)
    dom_data = ar.get(domain, {})
    for cfg in ar_order:
        if cfg not in dom_data:
            print(f"{cfg:<25} {'--':>12}")
            continue
        s = summarize(dom_data[cfg])
        if s is None:
            print(f"{cfg:<25} {'--':>12}")
            continue
        print(f"{cfg:<25} {fmt(*s['f1']):>12} {fmt(*s['on']):>12} {fmt(*s['ap']):>12} {fmt(*s['dur']):>10}")

print("\n" + "="*100)
print("Phase 1 Subtask Ablation -> Phase 2 (WLEL)")
print("="*100)

gp1 = data['gp1_wlel']
gp1_order = ['G0_full', 'GP1a_wo_has_apex', 'GP1b_wo_distance', 'GP1c_wo_phase', 'GP1d_wo_offset']
print(f"{'Config':<25} {'F1':>12} {'OnMAE':>12} {'ApMAE':>12} {'IntMAPE':>14}")
print("-"*80)
for cfg in gp1_order:
    if cfg not in gp1:
        print(f"{cfg:<25} {'--':>12}")
        continue
    s = summarize(gp1[cfg])
    if s is None:
        print(f"{cfg:<25} {'--':>12}")
        continue
    print(f"{cfg:<25} {fmt(*s['f1']):>12} {fmt(*s['on']):>12} {fmt(*s['ap']):>12} {fmtm(*s['mape']):>14}")
