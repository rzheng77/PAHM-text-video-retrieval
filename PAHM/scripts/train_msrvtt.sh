#batchsize-128  batch-size-val 32
export CUDA_VISIBLE_DEVICES="1"
export CUDA_LAUNCH_BLOCKING="1"
DATA_PATH=${HOME}/dataset
python -m torch.distributed.run --nproc_per_node=1\
    main_my.py --do_train --num_thread_reader=8 \
    --lr 1e-4 --batch_size=64  --batch_size_val 16 \
    --epochs=5  --n_display=100 \
    --train_csv ${DATA_PATH}/msrvtt/msrvtt_data/MSRVTT_train.9k.csv \
    --val_csv ${DATA_PATH}/msrvtt/msrvtt_data/MSRVTT_JSFUSION_test.csv \
    --data_path ${DATA_PATH}/msrvtt/msrvtt_data/MSRVTT_data.json \
    --features_path ${DATA_PATH}/msrvtt/resized_video \
    --output_dir ${HOME}/work11-2/output \
    --max_words 32 --max_frames 12 \
    --datatype msrvtt --expand_msrvtt_sentences  \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32