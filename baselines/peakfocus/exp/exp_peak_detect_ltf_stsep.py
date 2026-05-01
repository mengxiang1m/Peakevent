import os

import numpy as np
import torch

from data_provider.data_factory_stsep import data_provider_stsep
from exp.exp_peak_detect_based_on_long_term_forecasting import Exp_Peak_Detect_LTF


class Exp_Peak_Detect_LTF_STSEP(Exp_Peak_Detect_LTF):
    """PeakDetect experiment variant with STSEP-aligned data loading and exports."""

    def _build_model(self):
        # Force model to use 'peak_detect_ltf' so its forward() returns
        # the dual-head (value, peak) output instead of None.
        original_task = self.args.task_name
        self.args.task_name = 'peak_detect_ltf'
        model = super()._build_model()
        self.args.task_name = original_task
        return model

    def _get_data(self, flag):
        return data_provider_stsep(self.args, flag)

    def test(self, setting, test=0):
        # Keep original PeakFocus evaluation pipeline unchanged.
        super().test(setting, test)

        # Second pass: collect per-window outputs for STSEP post-hoc evaluation.
        test_data, test_loader = self._get_data('test')
        self.model.eval()

        preds_list = []
        peak_probs_list = []
        f_dim = -1 if self.args.features == 'MS' else 0

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) < 4:
                    raise RuntimeError('Expected at least 4 tensors in each batch')

                batch_x, batch_y, batch_x_mark, batch_y_mark = batch[:4]
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).to(self.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1)

                model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if not (isinstance(model_out, tuple) and len(model_out) == 2):
                    raise RuntimeError('Expected model output as (value_outputs, peak_outputs)')

                outputs, peak_outputs = model_out
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                peak_outputs = peak_outputs[:, -self.args.pred_len:, f_dim:]

                preds_list.append(outputs.detach().cpu().numpy())
                peak_probs_list.append(torch.sigmoid(peak_outputs).detach().cpu().numpy())

        if preds_list:
            preds_all = np.concatenate(preds_list, axis=0)
            peaks_all = np.concatenate(peak_probs_list, axis=0)
        else:
            preds_all = np.empty((0, self.args.pred_len, 1), dtype=np.float32)
            peaks_all = np.empty((0, self.args.pred_len, 1), dtype=np.float32)

        preds_all = test_data.inverse_transform(preds_all)

        result_dir = os.path.join('./results/', setting)
        os.makedirs(result_dir, exist_ok=True)

        np.save(os.path.join(result_dir, 'per_window_preds.npy'), preds_all.squeeze(-1))
        np.save(os.path.join(result_dir, 'per_window_peak_probs.npy'), peaks_all.squeeze(-1))
        np.save(os.path.join(result_dir, 'per_window_starts.npy'), test_data.starts_global)
        print(f"[STSEP] Saved per_window files to {result_dir}")
