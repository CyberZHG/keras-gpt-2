import os
import tempfile
from unittest import TestCase
import numpy as np
from keras_gpt_2.backend import keras
from keras_gpt_2 import get_model, get_custom_objects


class TestModel(TestCase):

    def test_save_load(self):
        model = get_model(
            n_vocab=50257,
            n_ctx=1024,
            n_embd=768,
            n_head=12,
            n_layer=12,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_gpt_2_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()

    def test_fixed_input_shape(self):
        model = get_model(
            n_vocab=50257,
            n_ctx=1024,
            n_embd=768,
            n_head=12,
            n_layer=12,
            fixed_input_shape=True,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_gpt_2_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects=get_custom_objects())
        model.summary()
