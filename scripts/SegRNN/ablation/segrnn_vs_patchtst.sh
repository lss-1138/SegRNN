
random_seed=2024
seq_len=720
pred_len=720

model_name=SegRNN

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --seg_len 48 \
  --enc_in 7 \
  --d_model 512 \
  --dropout 0.5 \
  --train_epochs 1 \
  --patience 1 \
  --rnn_type gru \
  --dec_way pmf \
  --channel_id 1 \
  --test_flop \
  --itr 1 --batch_size 8 --learning_rate 0.0002

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 48 \
      --enc_in 21 \
      --d_model 512 \
      --dropout 0.5 \
      --train_epochs 1 \
      --patience 1 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --test_flop \
      --itr 1 --batch_size 8 --learning_rate 0.0001

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --seg_len 48 \
      --enc_in 321 \
      --d_model 512 \
      --dropout 0.1 \
      --train_epochs 1 \
      --patience 1 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 1 \
      --test_flop \
      --itr 1 --batch_size 8 --learning_rate 0.0005




model_name=PatchTST


root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name_$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --fc_dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 1\
  --patience 1\
  --lradj 'TST'\
  --pct_start 0.4\
  --test_flop \
  --itr 1 --batch_size 8 --learning_rate 0.0001

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
python -u run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name_$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --e_layers 3 \
  --n_heads 16 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2\
  --fc_dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 1\
  --patience 1 \
  --test_flop \
  --itr 1 --batch_size 8 --learning_rate 0.0001

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom
python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name_$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --des 'Exp' \
    --train_epochs 1\
    --patience 1 \
    --lradj 'TST'\
    --pct_start 0.2 \
    --test_flop \
    --itr 1 --batch_size 8 --learning_rate 0.0001