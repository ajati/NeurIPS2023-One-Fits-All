export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/data/dnn_forecasting/FM/sota_evals/one_fits_all_fork/NeurIPS2023-One-Fits-All/Few-shot_Learning:$PYTHONPATH

seq_len=96
model=GPT4TS
pred_len=24
label_len=0
batch_size=256
patch_size=16
stride=8
percent=100 #not used here
data=service
data_path=Service.csv

python main.py \
    --root_path /data/dnn_forecasting/FM/ijcai24_target/data/bizitops \
    --data_path ${data_path} \
    --model_id ${data}_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ${data} \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --batch_size $batch_size \
    --learning_rate 0.001 \
    --train_epochs 30 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --freq 0 \
    --patch_size $patch_size \
    --stride $stride \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --is_gpt 1 2>&1 | tee exog_expts_logs/${data}_model_${model}_percent_${percent}_seq_${seq_len}_pred_${pred_len}.txt

