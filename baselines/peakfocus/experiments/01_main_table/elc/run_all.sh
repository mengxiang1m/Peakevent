#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."  # -> Code/
model_name=CycleNet
peak_tolerance=1
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --cycle 168 \
  --model_type mlp \
  --use_revin 1 \
  --d_model 64 \
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
  --peak_lookahead 5 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --cycle 168 \
  --model_type mlp \
  --use_revin 1 \
  --d_model 64 \
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
  --peak_lookahead 5 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --cycle 168 \
  --model_type mlp \
  --use_revin 1 \
  --d_model 64 \
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
  --peak_lookahead 5 \
  --gpu $gpu


python run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --cycle 168 \
  --model_type mlp \
  --use_revin 1 \
  --d_model 64 \
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
  --peak_lookahead 5 \
  --gpu $gpu

# model_name=PatchTST
# peak_tolerance=1
# gpu=0

# python run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --n_heads 4 \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --peak_lookahead 5 \
#   --gpu $gpu \
#   --itr 5

# python run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --n_heads 4 \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --peak_lookahead 5 \
#   --gpu $gpu \
#   --itr 5

# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --n_heads 4 \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --peak_lookahead 5 \
#   --gpu $gpu \
#   --itr 5

# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --n_heads 4 \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --peak_lookahead 5 \
#   --gpu $gpu \
#   --itr 5

# model_name=SegRNN
# peak_tolerance=1
# gpu=0


# python run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --seg_len 24 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --d_model 32 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu



# python run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --seg_len 24 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --d_model 32 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu

# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --seg_len 24 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --d_model 32 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu

# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --seg_len 24 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --d_model 32 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu

model_name=STID
peak_tolerance=1
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --num_nodes 1 \
  --node_dim 8 \
  --embed_dim 8 \
  --num_layer 1 \
  --temp_dim_tid 8 \
  --temp_dim_diw 8 \
  --time_of_day_size 24 \
  --day_of_week_size 7 \
  --if_T_i_D 1 \
  --if_D_i_W 1 \
  --if_node 0 \
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
  --peak_lookahead 5 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --num_nodes 1 \
  --node_dim 8 \
  --embed_dim 8 \
  --num_layer 1 \
  --temp_dim_tid 8 \
  --temp_dim_diw 8 \
  --time_of_day_size 24 \
  --day_of_week_size 7 \
  --if_T_i_D 1 \
  --if_D_i_W 1 \
  --if_node 0 \
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
  --peak_lookahead 5 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --num_nodes 1 \
  --node_dim 8 \
  --embed_dim 8 \
  --num_layer 1 \
  --temp_dim_tid 8 \
  --temp_dim_diw 8 \
  --time_of_day_size 24 \
  --day_of_week_size 7 \
  --if_T_i_D 1 \
  --if_D_i_W 1 \
  --if_node 0 \
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
  --peak_lookahead 5 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --num_nodes 1 \
  --node_dim 8 \
  --embed_dim 8 \
  --num_layer 1 \
  --temp_dim_tid 8 \
  --temp_dim_diw 8 \
  --time_of_day_size 24 \
  --day_of_week_size 7 \
  --if_T_i_D 1 \
  --if_D_i_W 1 \
  --if_node 0 \
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
  --peak_lookahead 5 \
  --gpu $gpu

# model_name=Transformer
# peak_tolerance=1
# gpu=0

# python run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --n_heads 2 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu

# python run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --n_heads 2 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu

# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --n_heads 2 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu

# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 256 \
#   --d_ff 512 \
#   --n_heads 2 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu


