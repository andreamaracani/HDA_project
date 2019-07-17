python3 src/train.py \
--datasetpath="data/" \
--architecture="cnn-trad-fpool3" \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \
--frame_length=800 \
--frame_step=500 \
--target_frame_number=38 \
--exclude_augmentation=False \

