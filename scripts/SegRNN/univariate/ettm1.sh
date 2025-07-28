model_name=SegRNN
seq_len=720

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1




python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'96 \
  --model $model_name \
  --data $data_name \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --seg_len 48 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --train_epochs 30 \
  --patience 5 \
  --rnn_type gru \
  --dec_way pmf \
  --channel_id 0 \
  --revin 0 \
  --itr 1 --batch_size 256 --learning_rate 0.0003

python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'192 \
  --model $model_name \
  --data $data_name \
  --features S \
  --seq_len $seq_len \
  --pred_len 192 \
  --seg_len 48 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --train_epochs 30 \
  --patience 5 \
  --rnn_type gru \
  --dec_way pmf \
  --channel_id 0 \
  --revin 0 \
  --itr 1 --batch_size 256 --learning_rate 0.0003


python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'336 \
  --model $model_name \
  --data $data_name \
  --features S \
  --seq_len $seq_len \
  --pred_len 336 \
  --seg_len 48 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --train_epochs 30 \
  --patience 5 \
  --rnn_type gru \
  --dec_way pmf \
  --channel_id 0 \
  --revin 0 \
  --itr 1 --batch_size 256 --learning_rate 0.0003

python -u run_longExp.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'720 \
  --model $model_name \
  --data $data_name \
  --features S \
  --seq_len $seq_len \
  --pred_len 720 \
  --seg_len 48 \
  --enc_in 1 \
  --d_model 512 \
  --dropout 0.5 \
  --train_epochs 30 \
  --patience 5 \
  --rnn_type gru \
  --dec_way pmf \
  --channel_id 0 \
  --revin 1 \
  --itr 1 --batch_size 256 --learning_rate 0.0003

