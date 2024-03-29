import subprocess

file = 'python src/train.py '

commands =  '--datasetpath="data/" ' \
            '--architecture="cnn-trad-fpool3" ' \
            '--filters 32 64 128 256 128 ' \
            '--kernel 3 3 ' \
            '--stride 1 1 ' \
            '--pool 2 2 ' \
            '--hidden_layers=5 ' \
            '--dropout_prob=0.3 ' \
            '--batchsize=64 ' \
            '--ckp_file="models/checkpoints/ckp" ' \
            '--num_epochs=10 ' \


subprocess.call(file + commands, shell=True)


