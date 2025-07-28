model_name=SegRNN

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

seq_len=720
for pred_len in 96 192 336 720
do
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
      --train_epochs 30 \
      --patience 5 \
      --rnn_type gru \
      --dec_way pmf \
      --channel_id 0 \
      --itr 1 --batch_size 256 --learning_rate 0.0003
done
