python -m torch.distributed.run --nproc_per_node=1\
    main_my.py --do_train --num_thread_reader=8 \
    --lr 1e-4 --batch_size=64  --batch_size_val 16 \
    --epochs=5  --n_display=100 \
    --data_path ${DATA_PATH}/msvd/msvd_data \
    --features_path ${DATA_PATH}/msvd/resized_video_3fps \
    --output_dir ${HOME}/work11-2/output/msvd \
    --max_words 32 --max_frames 12 \
    --datatype msvd \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0  --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/32


