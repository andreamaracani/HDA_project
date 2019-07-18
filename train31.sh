python3 src/train.py \
--datasetpath="data/" \
--name="train31_" \
--architecture="improved_cnn_trad_fpool3" \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \
--dropout_prob=0 \
--frame_length=400 \
--frame_step=160 \
--target_frame_number=110 \
--add_delta=False \
--normalization_method=0 \
--exclude_augmentation=True \
    