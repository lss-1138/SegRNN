if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=SegRNN

root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1


seq_len=192
pred_len=192
for random_seed in 2023 2024 2025 2026 2027
    do
    for seg_len in 192 96 48 24 12 6 1
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
          --seg_len $seg_len \
          --enc_in 7 \
          --d_model 512 \
          --dropout 0.5 \
          --train_epochs 30 \
          --patience 5 \
          --rnn_type gru \
          --dec_way pmf \
          --channel_id 1 \
          --random_seed $random_seed \
          --itr 1 --batch_size 256 --learning_rate 0.0002
    done
done

