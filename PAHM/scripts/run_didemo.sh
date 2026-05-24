# nohup sh scripts/run_didemo.sh > /home/raozheng/work11-2/output/didemo/sh.log 2>&1 &
#batchsize-128  batch-size-val 32
export CUDA_VISIBLE_DEVICES=1,2 \
# export CUDA_LAUNCH_BLOCKING="1"
DATA_PATH=${HOME}/dataset
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29553\
    main_my.py --do_train  --num_thread_reader=8 \
    --epochs=5 --batch_size=24 --n_display=50 \
    --data_path ${DATA_PATH}/didemo/didemo_data \
    --features_path ${DATA_PATH}/didemo/DiDeMo_resized\
    --output_dir /home/raozheng/work11-2/output/didemo \
    --lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 8 \
    --datatype didemo \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 

