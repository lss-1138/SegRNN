if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=VanillaRNN

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=96
for pred_len in 1 2 4 8 16 32 64 128 256 512
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
      --enc_in 7 \
      --d_model 256 \
      --train_epochs 30 \
      --patience 5 \
      --rnn_type rnn \
      --des rnn \
      --itr 5 --batch_size 64 --learning_rate 0.001
done

for pred_len in 1 2 4 8 16 32 64 128 256 512
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
      --enc_in 7 \
      --d_model 256 \
      --train_epochs 30 \
      --patience 5 \
      --rnn_type lstm \
      --des lstm \
      --itr 5 --batch_size 64 --learning_rate 0.001
done

for pred_len in 1 2 4 8 16 32 64 128 256 512
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
      --enc_in 7 \
      --d_model 256 \
      --train_epochs 30 \
      --patience 5 \
      --rnn_type gru \
      --des gru \
      --itr 5 --batch_size 64 --learning_rate 0.001
done
