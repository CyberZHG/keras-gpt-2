import os
from unittest import TestCase
from keras_gpt_2 import load_trained_model_from_checkpoint


class TestLoader(TestCase):

    def test_load_from_checkpoint(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        toy_checkpoint_path = os.path.join(current_path, 'toy_checkpoint')
        config_path = os.path.join(toy_checkpoint_path, 'hparams.json')
        checkpoint_path = os.path.join(toy_checkpoint_path, 'model.ckpt')
        model = load_trained_model_from_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
        model.summary()

    def test_load_from_checkpoint_shorter(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        toy_checkpoint_path = os.path.join(current_path, 'toy_checkpoint')
        config_path = os.path.join(toy_checkpoint_path, 'hparams.json')
        checkpoint_path = os.path.join(toy_checkpoint_path, 'model.ckpt')
        model = load_trained_model_from_checkpoint(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            seq_len=10,
        )
        model.summary()
