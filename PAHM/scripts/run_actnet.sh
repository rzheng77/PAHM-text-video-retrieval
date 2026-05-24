DATA_PATH=${HOME}/dataset
python -m torch.distributed.launch --nproc_per_node=1 \
    main_xclip.py --do_train --num_thread_reader=4 \
    --epochs=20 --batch_size=64 --n_display=50 \
    --data_path ${DATA_PATH}/ActivityNet \
    --features_path ${DATA_PATH}/ActivityNet/Activity_Videos \
    --output_dir ckpts_dsw/${job_name} \
    --lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 16 \
    --datatype activity \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32 
