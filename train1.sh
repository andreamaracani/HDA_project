python3 src/train.py \
--datasetpath="data/" \
--architecture="cnn-trad-fpool3" \
--dropout_prob=0.3 \
--add_delta=False \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \