"""Benchmark: 1-batch forward+backward speed for independent_dense_int mode.
Compares old (CPU decode + Python loop) vs new (GPU vectorised) path.
"""
import sys, os, time, argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)  # TODO(R-future): migrate to patchevent package import

from patchevent.phase2.dataset import build_dataloaders
from patchevent.phase2.model import build_model

# ---------- config ----------
ENCODER_CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'phase1', 'checkpoints', 'wlel', 's42', 'best_model.pth')
SERIES = r'd:\ywz_experiment_papers\ywz_7_electric\ywz2\dataset\wlel\event_v1\data\wlel_event_series_v1.csv'
EVENTS = r'd:\ywz_experiment_papers\ywz_7_electric\ywz2\dataset\wlel\event_v1\data\wlel_events_v1.jsonl'

def make_args():
    a = argparse.Namespace(
        encoder_ckpt=ENCODER_CKPT,
        d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.2,
        max_seq_len=160, pred_len=336, event_schema='triplet',
        encoder_mode='frozen', embed_dropout=0.15,
        unfreeze_last_n=0, use_memory_pos=False, use_self_attn_agg=False,
        use_input_decomp=False, input_decomp_mode='full', use_raw_bypass=False,
        use_tuple_guided_value=False, use_independent_dense_int=True,
        use_apex_cond_int=False, use_int_direct_lookup=False,
        use_decoupled_int_head=False, use_tuple_crossatt_int=False,
        min_onset_spacing=0, override_norm_mean=None, override_norm_std=None,
        no_pos_valid_mask=False, no_causal_mask=False,
        use_int_regression=False, use_series_stats=False,
        series_stats_recent_k=8, intensity_from_values='apex',
    )
    return a

def bench_forward_backward(model, batch, device, n_warmup=3, n_iter=10, tag=''):
    x = batch['x'].to(device)
    x_future = batch['x_future'].to(device)
    target_ids = batch['target_ids'].to(device)

    # warmup
    for _ in range(n_warmup):
        out = model(x, target_ids)
        logits = out['logits'] if isinstance(out, dict) else out
        fv = out.get('future_values_norm') if isinstance(out, dict) else None
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1),
            ignore_index=0,
        )
        if fv is not None:
            gt = (x_future.float() - model.norm_mean) / model.norm_std
            loss = loss + torch.nn.functional.mse_loss(fv, gt)
        loss.backward()
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = model(x, target_ids)
        logits = out['logits'] if isinstance(out, dict) else out
        fv = out.get('future_values_norm') if isinstance(out, dict) else None
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_ids[:, 1:].reshape(-1),
            ignore_index=0,
        )
        if fv is not None:
            gt = (x_future.float() - model.norm_mean) / model.norm_std
            loss = loss + torch.nn.functional.mse_loss(fv, gt)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter
    print(f'[{tag}] {elapsed*1000:.1f} ms/batch  ({n_iter} iters)')
    return elapsed

def main():
    device = torch.device('cuda:0')
    args = make_args()

    print('--- Loading data (1 batch) ---')
    train_loader, _, _, meta = build_dataloaders(
        series_path=SERIES, events_path=EVENTS,
        seq_len=96, pred_len=336, event_schema='triplet',
        batch_size=64, eval_batch_size=64,
    )
    batch = next(iter(train_loader))
    print(f'batch x={batch["x"].shape} target_ids={batch["target_ids"].shape}')

    print('\n--- Building model (GPU path = current code) ---')
    model = build_model(args).to(device)
    model.train()

    t_gpu = bench_forward_backward(model, batch, device, tag='GPU-vectorised')

    # --- Old path: monkey-patch forward to use CPU decode ---
    print('\n--- Monkey-patching model.forward to use OLD CPU path ---')
    original_forward = model.forward.__func__

    def old_forward(self, x, target_ids, **kwargs):
        if self.use_apex_cond_int or self.use_independent_dense_int:
            memory, x_norm = self._encode_patches(x, return_x_norm=True)
        else:
            memory = self._encode_patches(x)
        decoder_input = target_ids[:, :-1]
        T = decoder_input.size(1)
        positions = torch.arange(T, device=x.device)
        tgt = self.embed_drop(self.token_embed(decoder_input) + self.pos_embed(positions))
        causal_mask = None if self.no_causal_mask else self._make_causal_mask(T, x.device)
        tgt_pad_mask = (decoder_input == self.tokenizer.PAD_ID)
        out = self.decoder(tgt, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_pad_mask)
        hidden = self.out_drop(out)
        logits = self.lm_head(hidden)
        if not self.no_pos_valid_mask and T <= self.max_seq_len:
            pos_mask = self.pos_valid_mask[:T]
            logits = logits.masked_fill(~pos_mask.unsqueeze(0), float('-inf'))
        if self.use_independent_dense_int:
            events_batch = self._events_from_token_batch(target_ids)
            future_values_norm, _ = self._predict_independent_future_values_norm(x_norm, events_batch)
            return {
                'logits': logits,
                'future_values_norm': future_values_norm,
                'events_batch': events_batch,
            }
        return logits

    import types
    model.forward = types.MethodType(old_forward, model)
    t_cpu = bench_forward_backward(model, batch, device, tag='OLD-CPU-decode')

    print(f'\n=== Speedup: {t_cpu/t_gpu:.1f}x ({t_cpu*1000:.1f}ms → {t_gpu*1000:.1f}ms) ===')

if __name__ == '__main__':
    main()
