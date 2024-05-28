# SimCLNMT: A Simple Contrastive Learning Method for Enhancing Neural Machine Translation Quality
- train.sh

envs_dir=
export PATH=$envs_dir/bin:$PATH
base_dir=
codes_dir=$base_dir/codes/fairseq
datas=$1
seed=1111
max_tokens=4096
dropout=0.1
attention_heads=8
embed_dim=512
ffn_embed_dim=2048
encoder_layer=6
decoder_layer=6
src=en
tgt=ch
train_dataset_dir=
export CUDA_VISIBLE_DEVICES=0,1,2,3
hf_pen=0
warmup_updates=4000
ratio=1
lr=2e-04
update_freq=2
echo "begin lr ${lr}"
save_dir=$base_dir/
mkdir -p $save_dir
echo "begin training based on hf_len is ${hf_pen} batch_size is ${batch_size}"
python $codes_dir/train.py $train_dataset_dir \
     --task translation_rrhf \
     --arch hf_transformer_base \
     --restore-file $base_dir/models/checkpoints-nist-en-ch-subword-32000-test/average.pt \
     --dataset-impl='raw_json' \
     --source-lang $src \
     --target-lang $tgt \
     --save-dir $save_dir \
     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
     --lr-scheduler inverse_sqrt --lr ${lr} --max-epoch 15 \
     --reset-optimizer \
     --reset-dataloader \
     --disable-validation \
     --batch-size 8 \
     --fp16 \
     --num-workers 0 \
     --warmup-updates ${warmup_updates} --warmup-init-lr '1e-07' \
     --criterion hf_label_smoothed_cross_entropy --label-smoothing 0.1 --hf-score-penalty ${hf_pen} --hf-loss-ratio ${ratio}\
     --seed ${seed} \
     --max-tokens ${max_tokens} --update-freq ${update_freq} \
     --dropout ${dropout} --relu-dropout 0.1 --attention-dropout 0.1 \
     --decoder-attention-heads ${attention_heads} --encoder-attention-heads ${attention_heads} \
     --decoder-embed-dim ${embed_dim} --encoder-embed-dim ${embed_dim} \
     --decoder-ffn-embed-dim ${ffn_embed_dim} --encoder-ffn-embed-dim ${ffn_embed_dim} \
     --encoder-layers ${encoder_layer} --decoder-layers ${decoder_layer}
