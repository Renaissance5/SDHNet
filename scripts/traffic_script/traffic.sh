#traffic_720_96
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/traffic \
  --data_path traffic.csv \
  --model_id traffic_720_96 \
  --model SDHNet \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --train_epochs 10 \
  --dropout 0.05 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --d_model 512 \
  --activation tanh \
  --pooling_size 8 \
  --period 24 \
  --enc_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1

#traffic_720_192
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/traffic \
  --data_path traffic.csv \
  --model_id traffic_720_192 \
  --model SDHNet \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 192 \
  --train_epochs 10 \
  --dropout 0.05 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --d_model 512 \
  --activation tanh \
  --pooling_size 8 \
  --period 24 \
  --enc_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1

#traffic_720_336
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/traffic \
  --data_path traffic.csv \
  --model_id traffic_720_336 \
  --model SDHNet \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 336 \
  --train_epochs 10 \
  --dropout 0.05 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --d_model 512 \
  --activation tanh \
  --pooling_size 8 \
  --period 24 \
  --enc_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1

#traffic_720_720
python -u run.py \
  --is_training 1 \
  --root_path ./datasets/traffic \
  --data_path traffic.csv \
  --model_id traffic_720_720 \
  --model SDHNet \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --train_epochs 10 \
  --dropout 0.05 \
  --learning_rate 0.0005 \
  --batch_size 16 \
  --d_model 512 \
  --activation tanh \
  --pooling_size 8 \
  --period 24 \
  --enc_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --itr 1