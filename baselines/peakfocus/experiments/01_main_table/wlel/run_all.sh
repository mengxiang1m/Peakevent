#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."  # -> Code/
model_name=CycleNet
peak_tolerance=1
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --cycle 168 \
  --model_type mlp \
  --use_revin 1 \
  --d_model 1024 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --cycle 168 \
  --model_type mlp \
  --use_revin 1 \
  --d_model 1024 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu


model_name=MetaEformer
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --d_model 256 \
  --d_ff 512 \
  --d_low 10 \
  --n_heads 2 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 1 \
  --enable_peak_eval 1 \
  --peak_tolerance 1 \
  --freq h \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --mpp_update 50 \
  --sim_num 10 \
  --threshold 1 \
  --mpp_size 350 \
  --mp_len 24 \
  --kernel_size 24 \
  --dim_static 0 \
  --if_padding 1 \
  --output_attention 0 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --d_model 256 \
  --d_ff 512 \
  --d_low 10 \
  --n_heads 2 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance 1 \
  --freq h \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --mpp_update 50 \
  --sim_num 10 \
  --threshold 1 \
  --mpp_size 350 \
  --mp_len 24 \
  --kernel_size 24 \
  --dim_static 0 \
  --if_padding 1 \
  --output_attention 0 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu

model_name=SegRNN
peak_tolerance=1
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --seg_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --seg_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu

model_name=TimeMixer
peak_tolerance=1
gpu=0


python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu


python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --down_sampling_layers 3 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu


model_name=DLinear
peak_tolerance=1
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 3 \
  --gpu $gpu

model_name=PatchTST
peak_tolerance=1
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --n_heads 4 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --peak_lookahead 3 \
  --gpu $gpu \
  --itr 5

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --n_heads 4 \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --peak_lookahead 3 \
  --gpu $gpu \
  --itr 5
