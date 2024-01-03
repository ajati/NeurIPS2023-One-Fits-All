export CUDA_VISIBLE_DEVICES=0

seq_len=512
model=GPT4TS
percent=10
pred_len=96
lr=0.0001

python main.py \
    --root_path /dccstor/tsfm-irl/vijaye12/datasets/public/ETDataset/ETT-small \
    --data_path ETTh1.csv \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ett_h \
    --features M \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 64 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 1 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --freq 0 \
    --patch_size 16 \
    --stride 16 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 2>&1 | tee log_etth1_model_${model}_percent_${percent}_seq_${seq_len}_pred_${pred_len}.txt