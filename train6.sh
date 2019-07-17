python3 src/train.py \
--datasetpath="data/" \
--architecture="cnn-trad-fpool3" \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \
--dropout_prob=0 \
--frame_length=400 \
--frame_step=160 \
--target_frame_number=110 \
--add_delta=True \
--normalization_method=1 \
--exclude_augmentation=True \
    