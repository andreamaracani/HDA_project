python3 src/train.py \
--datasetpath="data/" \
--name="train73_" \
--architecture="improved-cnn-trad-fpool3" \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \
--dropout_prob=0 \
--frame_length=800 \
--frame_step=500 \
--target_frame_number=38 \
--add_delta=False \
--normalization_method=0 \
--exclude_augmentation=True \
    