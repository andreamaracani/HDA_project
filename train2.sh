python3 src/train.py \
--datasetpath="data/" \
--architecture="cnn-trad-fpool3" \
--normalization_method=2
--dropout_prob=0.3 \
--batchsize=64 \
--ckp_folder="checkpoints/" \
--num_epochs=20 \
