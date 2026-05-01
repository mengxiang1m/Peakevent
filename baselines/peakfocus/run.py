import argparse
import os
import torch
import torch.backends
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_peak_detect_based_on_long_term_forecasting import Exp_Peak_Detect_LTF
from exp.exp_peak_detect_based_on_long_term_forecasting_basic import Exp_Peak_Detect_LTF_basic
from exp.exp_peak_detect_based_on_long_term_forecasting_seq2peak import Exp_Peak_Detect_LTF_Seq2Peak
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, peak_detect_ltf, peak_detect_ltf_basic, seq2peak, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETTh1/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--input_col', type=str, default='value_max', help='input column name for mixed dataloader')
    parser.add_argument('--target_col', type=str, default='value_max', help='target column name for mixed dataloader')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--output_attention', type=int, default=0, help='whether to output attention; True 1 False 0')

    # CycleNet
    parser.add_argument('--cycle', type=int, default=168, help='cycle length for CycleNet (168 for weekly cycle in hourly data)')
    parser.add_argument('--model_type', type=str, default='mlp', help='model type for CycleNet, options:[linear, mlp]')
    parser.add_argument('--use_revin', type=int, default=1, help='whether to use RevIN normalization for CycleNet; True 1 False 0')

    # STID
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes for STID')
    parser.add_argument('--node_dim', type=int, default=32, help='node embedding dimension for STID')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension for STID')
    parser.add_argument('--num_layer', type=int, default=3, help='number of layers for STID')
    parser.add_argument('--temp_dim_tid', type=int, default=32, help='time of day dimension for STID')
    parser.add_argument('--temp_dim_diw', type=int, default=32, help='day of week dimension for STID')
    parser.add_argument('--time_of_day_size', type=int, default=24, help='time of day size for STID')
    parser.add_argument('--day_of_week_size', type=int, default=7, help='day of week size for STID')
    parser.add_argument('--if_T_i_D', type=int, default=1, help='whether to use time in day for STID')
    parser.add_argument('--if_D_i_W', type=int, default=1, help='whether to use day in week for STID')
    parser.add_argument('--if_node', type=int, default=0, help='whether to use spatial embeddings for STID')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=5, help='experiments times')
    parser.add_argument('--start_seed', type=int, default=0,
                        help='starting iteration index (ii) for the itr loop; used to fill missing seed slots without clobbering existing ones')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', 
                        help='loss function: MSE, MAE, Wasserstein, Energy, Combined_MSE_Wass, Combined_MAE_Wass, Combined_MSE_Energy')
    parser.add_argument('--loss_alpha', type=float, default=0.5, 
                        help='weight for statistical loss in combined loss (0-1), loss = (1-alpha)*MSE + alpha*Wasserstein')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', '--patch_lens', dest='patch_len', type=int, default=16, help='patch length')
    
    # ===================================================================================================================
    # ================================================MetaEformer========================================================
    # ===================================================================================================================
    # MetaEformer specific parameters
    parser.add_argument('--d_low', type=int, default=10, help='dimension of low-rank representation for MetaEformer')
    parser.add_argument('--mpp_update', type=int, default=50, help='meta pattern pool update frequency for MetaEformer')
    parser.add_argument('--sim_num', type=int, default=10, help='number of similar patterns to use for MetaEformer')
    parser.add_argument('--threshold', type=float, default=1.0, help='threshold for pattern matching in MetaEformer')
    parser.add_argument('--mpp_size', type=int, default=350, help='meta pattern pool size for MetaEformer')
    parser.add_argument('--mp_len', type=int, default=96, help='meta pattern length for MetaEformer')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size for series decomposition in MetaEformer')
    parser.add_argument('--dim_static', type=int, default=0, help='dimension of static context for MetaEformer')
    parser.add_argument('--if_padding', type=int, default=1, help='whether to use padding in MetaEformer (0 or 1)')
    
    # ===================================================================================================================
    # ==================================================Peak Detection===================================================
    # ===================================================================================================================
    # Peak Evaluation Control (which datasets to evaluate and visualize)
    parser.add_argument('--enable_peak_eval', type=int, default=1,
                        help='Enable peak detection evaluation during validation and testing (0 or 1)')
    parser.add_argument('--eval_train_peaks', type=int, default=1,
                        help='Evaluate peaks on training set (0 or 1)')
    parser.add_argument('--eval_val_peaks', type=int, default=1,
                        help='Evaluate peaks on validation set (0 or 1)')
    parser.add_argument('--eval_test_peaks', type=int, default=1,
                        help='Evaluate peaks on test set (0 or 1)')
    parser.add_argument('--vis_train_peaks', type=int, default=0,
                        help='Visualize peaks on training set (0 or 1)')
    parser.add_argument('--vis_val_peaks', type=int, default=0,
                        help='Visualize peaks on validation set (0 or 1)')
    parser.add_argument('--vis_test_peaks', type=int, default=1,
                        help='Visualize peaks on test set (0 or 1)')
    
    # Args for findpeaks method
    parser.add_argument('--peak_lookahead', type=int, default=3,
                        help='Lookahead parameter for findpeaks peak detection algorithm')
    parser.add_argument('--peak_method', type=str, default='peakdetect',
                        help='Peak detection method for findpeaks library, options: [peakdetect, topology, mask, etc.]')
    parser.add_argument('--peak_workers', type=int, default=8,
                        help='Number of workers for multiprocess peak detection')
    
    # Peak classification parameters (for peak_detect_ltf task)
    parser.add_argument('--peak_tolerance', type=int, default=1,
                        help='Tolerance window for peak matching (in time steps)')
    parser.add_argument('--peak_threshold', type=float, default=0.4,
                        help='Threshold for peak classification (0-1)')
    parser.add_argument('--use_soft_labels', type=int, default=1,
                        help='Use soft labels (Gaussian smoothing) for peak classification (0 or 1)')
    parser.add_argument('--soft_label_sigma', type=float, default=2.0,
                        help='Sigma for Gaussian smoothing in soft labels')
    parser.add_argument('--soft_label_tolerance', type=int, default=3,
                        help='Tolerance window for soft label smoothing (in time steps)')
    
    # Loss weights (for peak_detect_ltf task)
    parser.add_argument('--value_loss_weight', type=float, default=0.2,
                        help='Weight for value prediction loss')
    parser.add_argument('--peak_loss_weight', type=float, default=0.4,
                        help='Weight for peak classification loss')
    parser.add_argument('--tp_mse_loss_weight', type=float, default=0.4,
                        help='Weight for TP MSE loss (indexed loss on true peaks)')
    
    # Focal Loss parameters (for hard labels in peak classification)
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for Focal Loss (weight for positive class)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss (focusing parameter)')
    
    # ==================================================Seq2Peak===================================================
    # Parameters for peak_detect_ltf_seq2peak task (peak_Transformer model)
    parser.add_argument('--seq2peak_value_loss_weight', type=float, default=0.5,
                        help='Weight for full sequence value prediction loss in seq2peak')
    parser.add_argument('--seq2peak_peak_loss_weight', type=float, default=0.5,
                        help='Weight for peak value prediction loss in seq2peak')
    
    # ==================================================peak_Transformer Model Parameters===================================================
    parser.add_argument('--busy_decoder', type=int, default=0,
                        help='Enable busy decoder for peak detection (0 or 1)')
    parser.add_argument('--busy_decoder_modal', type=str, default='m',
                        help='Busy decoder mode: m for MaxPool, l for Linear')
    parser.add_argument('--hour_day', type=str, default='h',
                        help='Hour-day mode for peak_Transformer: h or d')
    parser.add_argument('--with_shift', type=str, default='ws',
                        help='With shift module: ws (with shift) or ns (no shift)')
    
    # ==================== Derivative Features Parameters ====================
    parser.add_argument('--mlp_layers', type=int, default=2,
                        help='Number of layers in the MLP backbone')
    parser.add_argument('--if_msm_pl', type=int, default=1,
                        help='If use Multi Scale Mixing Peak Locator (MSM-PL), (0 or 1)')
    parser.add_argument('--if_lad', type=int, default=1,
                        help='If use Location-Aware Decoder (LAD), (0 or 1)')

    # ==================== Naive Baseline Parameters ====================
    parser.add_argument('--naive_type', type=str, default='persistence',
                        help='Naive baseline type: persistence, seasonal, movavg')
    parser.add_argument('--movavg_window', type=int, default=24,
                        help='Window size for moving average naive baseline')
    parser.add_argument('--seasonal_period', type=int, default=168,
                        help='Seasonal period for seasonal naive baseline (default 168 = 7 days * 24 hours)')

    # ==================== Ablation Parameters ====================
    parser.add_argument('--mask_type', type=str, default='soft',
                        help='Peak mask type for loss: soft (Gaussian) or hard (binary)')
    parser.add_argument('--n_scales', type=int, default=2,
                        help='Number of scales in MSM-PL pyramid (0=no pyramid, 1=single, 2=default, 3=triple)')

    # ==================== External Features (Weather Covariates) ====================
    parser.add_argument('--use_external_features', type=int, default=0,
                        help='Enable external feature injection (e.g. weather covariates). 0=off, 1=on')
    parser.add_argument('--external_feature_cols', type=str, default='',
                        help='Comma-separated column names of external features in the CSV, e.g. "temperature,humidity"')
    parser.add_argument('--external_feature_mode', type=str, default='concat',
                        choices=['concat', 'fusion', 'timexer', 'future_only'],
                        help='How to use external features: concat uses direct concatenation, fusion uses history weather in encoder and future-known weather in decoder, timexer uses a separate exogenous bridge, future_only injects known future covariates only into decoder')

    args = parser.parse_args()
    ext_cols = [c.strip() for c in args.external_feature_cols.split(',') if c.strip()]
    args.num_external_features = len(ext_cols)
    args.endogenous_in = args.c_out
    if args.use_external_features and args.num_external_features == 0:
        raise ValueError('use_external_features=1 requires non-empty external_feature_cols')
    if args.use_external_features and args.model == 'proposed_model' and args.task_name in ['peak_detect_ltf', 'peak_detect_ltf_basic']:
        if args.external_feature_mode in ['concat', 'fusion']:
            expected_enc_in = args.endogenous_in + args.num_external_features
            expected_dec_in = expected_enc_in
        elif args.external_feature_mode == 'timexer':
            expected_enc_in = args.endogenous_in + args.num_external_features
            expected_dec_in = args.endogenous_in
        else:
            expected_enc_in = args.endogenous_in
            expected_dec_in = args.endogenous_in + args.num_external_features
        if args.enc_in != expected_enc_in:
            print(f'Overriding enc_in from {args.enc_in} to {expected_enc_in} for external feature mode {args.external_feature_mode}')
            args.enc_in = expected_enc_in
        if args.dec_in != expected_dec_in:
            print(f'Overriding dec_in from {args.dec_in} to {expected_dec_in} for external feature mode {args.external_feature_mode}')
            args.dec_in = expected_dec_in

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'peak_detect_ltf':
        Exp = Exp_Peak_Detect_LTF
    elif args.task_name == 'peak_detect_ltf_basic':
        Exp = Exp_Peak_Detect_LTF_basic
    elif args.task_name == 'seq2peak':
        Exp = Exp_Peak_Detect_LTF_Seq2Peak
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.start_seed, args.start_seed + args.itr):
            # setting record of experiments
            # Simplified setting name to avoid path length issues
            setting = '{}_{}_{}_{}_{}_{}_{}'.format(
                args.task_name,
                args.data,
                args.model,
                args.model_id,
                args.seq_len,
                args.pred_len,
                ii)
            
            # Add setting and iteration index to args
            args.setting = setting
            args.itr_index = ii
            exp = Exp(args)  # set experiments

            # Setup experiment log file
            results_folder = './results/' + setting + '/'
            if not os.path.exists(results_folder):
                os.makedirs(results_folder)
            
            log_file_path = results_folder + 'experiment_log.txt'
            from datetime import datetime
            
            # Initialize log file
            with open(log_file_path, 'w') as log_f:
                log_f.write("EXPERIMENT LOG\n")
                log_f.write("=" * 50 + "\n")
                log_f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_f.write(f"Setting: {setting}\n")
                log_f.write(f"Model: {args.model}\n")
                log_f.write(f"Dataset: {args.data}\n")
                log_f.write(f"Prediction: {args.seq_len}->{args.pred_len}\n")
                log_f.write(f"Train Epochs: {args.train_epochs}\n")
                log_f.write("\nTRAINING LOG\n")
                log_f.write("=" * 50 + "\n")
            
            # Pass log file path to experiment
            exp.log_file_path = log_file_path

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        ii = 0
        # Simplified setting name to avoid path length issues
        setting = '{}_{}_{}_{}_{}_{}_{}'.format(
            args.task_name,
            args.data,
            args.model,
            args.model_id,
            args.seq_len,
            args.pred_len,
            ii)
        
        # Add setting and iteration index to args
        args.setting = setting
        args.itr_index = ii
        exp = Exp(args)  # set experiments

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # NaiveBaseline不需要加载checkpoint，直接用随机初始化的模型（无可训练参数）
        load_ckpt = 0 if args.model == 'NaiveBaseline' else 1
        exp.test(setting, test=load_ckpt)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
