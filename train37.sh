python3 src/train.py \
--datasetpath="data/" \
--name="train37_" \
--architecture="improved_cnn_trad_fpool3" \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \
--dropout_prob=0 \
--frame_length=800 \
--frame_step=500 \
--target_frame_number=38 \
--add_delta=True \
--normalization_method=2 \
--exclude_augmentation=True \
    