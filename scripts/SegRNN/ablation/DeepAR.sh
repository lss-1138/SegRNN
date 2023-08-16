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
    --enc_in 7 \
    --d_model 128 \
    --train_epochs 30 \
    --patience 3 \
    --rnn_type lstm \
    --des lstm \
    --itr 1 --batch_size 256 --learning_rate 0.001
done

root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
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
    --enc_in 7 \
    --d_model 128 \
    --train_epochs 30 \
    --patience 3 \
    --rnn_type lstm \
    --des lstm \
    --itr 1 --batch_size 256 --learning_rate 0.001
done

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
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
    --enc_in 7 \
    --d_model 128 \
    --train_epochs 30 \
    --patience 3 \
    --rnn_type lstm \
    --des lstm \
    --itr 1 --batch_size 256 --learning_rate 0.001
done

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
    --enc_in 7 \
    --d_model 128 \
    --train_epochs 30 \
    --patience 3 \
    --rnn_type lstm \
    --des lstm \
    --itr 1 --batch_size 256 --learning_rate 0.001
done

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom
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
    --enc_in 21 \
    --d_model 128 \
    --train_epochs 30 \
    --patience 3 \
    --rnn_type lstm \
    --des lstm \
    --itr 1 --batch_size 128 --learning_rate 0.001
done

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
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
    --enc_in 321 \
    --d_model 128 \
    --train_epochs 30 \
    --patience 3 \
    --rnn_type lstm \
    --des lstm \
    --itr 1 --batch_size 64 --learning_rate 0.001
done

root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
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
    --enc_in 862 \
    --d_model 128 \
    --train_epochs 30 \
    --patience 3 \
    --rnn_type lstm \
    --des lstm \
    --itr 1 --batch_size 64 --learning_rate 0.001
done
