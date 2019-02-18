import os
from unittest import TestCase
from keras_gpt_2 import get_bpe


class TestBPE(TestCase):

    def test_encode_and_decode(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        toy_checkpoint_path = os.path.join(current_path, 'toy_checkpoint')
        encoder_path = os.path.join(toy_checkpoint_path, 'encoder.json')
        vocab_path = os.path.join(toy_checkpoint_path, 'vocab.bpe')
        bpe = get_bpe(encoder_path, vocab_path)
        text = 'Power, give me more power!'
        indices = bpe.encode(text)
        self.assertEqual([13434, 11, 1577, 502, 517, 1176, 0], indices)
        self.assertEqual(text, bpe.decode(indices))
        self.assertEqual(text, bpe.decode(bpe.encode(text)))
