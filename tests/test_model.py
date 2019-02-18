import os
import random
import tempfile
import keras
from unittest import TestCase
from keras_gpt_2 import get_model, get_custom_objects


class TestModel(TestCase):

    def test_save_load(self):
        model = get_model(
            n_vocab=50257,
            n_ctx=1024,
            n_embd=768,
            n_head=12,
            n_layer=12
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_gpt_2_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
