python3 src/train.py \
--datasetpath="data/" \
--name="train28_" \
--architecture="AttRNNSpeechModel" \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \
--dropout_prob=0 \
--frame_length=1024 \
--frame_step=128 \
--number_of_filters=80 \
--target_frame_number=135 \
--add_delta=False \
--normalization_method=0 \
--exclude_augmentation=True \
    