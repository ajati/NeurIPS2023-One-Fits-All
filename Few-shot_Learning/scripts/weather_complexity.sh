export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS
percent=10
pred_len=96
label_len=0
batch_size=64
patch_size=64
stride=64

python main.py \
    --root_path /dccstor/tsfm23/datasets/weather/ \
    --data_path weather.csv \
    --model_id weather_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --learning_rate 0.001 \
    --train_epochs 1 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --freq 0 \
    --lradj type3 \
    --patch_size $patch_size \
    --stride $stride \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --is_gpt 1 2>&1 | tee complexity_logs_tmp/complexity_weather_model_${model}_percent_${percent}_seq_${seq_len}_pred_${pred_len}.txt
    