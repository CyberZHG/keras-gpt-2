import os
import sys
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate


if len(sys.argv) != 2:
    print('python3 demo.py MODEL_FOLDER')
    sys.exit(-1)


model_folder = sys.argv[1]
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


print('Load model from checkpoint...')
model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
print('Load BPE from files...')
bpe = get_bpe_from_files(encoder_path, vocab_path)
print('Generate text...')
output = generate(model, bpe, ['From the day forth, my arm'], length=20, top_k=40)

# If you are using the 117M model and top_k equals to 1, then the result would be:
# "From the day forth, my arm was broken, and I was in a state of pain. I was in a state of pain,"
print(output[0])
