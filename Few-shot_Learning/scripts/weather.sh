export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS

for percent in 100
do
for pred_len in 96
do

python main.py \
    --root_path /dccstor/tsfm23/datasets/weather/ \
    --data_path weather.csv \
    --model_id weather_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.001 \
    --train_epochs 1 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --lradj type3 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --is_gpt 1 2>&1 | tee log_weather_model_${model}_percent_${percent}.txt
    
done
done