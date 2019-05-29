import os
from unittest import TestCase
import numpy as np
from keras_gpt_2 import get_model, BytePairEncoding, generate


try:
    chr = unichr
except Exception as e:
    '''No need to use `unichr` in Python 3'''


class TestGen(TestCase):

    def test_train_and_gen(self):
        token_dict = {chr(i): i for i in range(2 ** 9)}
        token_dict['Po'] = len(token_dict)
        token_dict['er'] = len(token_dict)
        model = get_model(
            n_vocab=len(token_dict),
            n_ctx=100,
            n_embd=30,
            n_head=5,
            n_layer=2,
        )
        bpe = BytePairEncoding(token_dict=token_dict, bpe_rank={('P', 'o'): 0, ('e', 'r'): 1})
        texts = [
            'Power, give me more power!',
            'From the day forth, my arm changed.',
        ]
        space_encode = bpe.encode(' ')
        inputs = [bpe.encode(text) for text in texts]
        max_len = max(map(len, inputs))
        inputs = [encode + space_encode * (max_len - len(encode)) for encode in inputs]
        outputs = [encode[1:] + space_encode for encode in inputs]
        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path, 'gen_test.h5')
        if os.path.exists(model_path):
            model.load_weights(model_path)
            model.fit(
                x=np.array(inputs * 1000),
                y=np.expand_dims(np.array(outputs * 1000), axis=-1),
                epochs=1,
            )
        else:
            model.fit(
                x=np.array(inputs * 1000),
                y=np.expand_dims(np.array(outputs * 1000), axis=-1),
                epochs=10,
            )
            model.save_weights(model_path)
        texts = [
            'Power, give me more',
            'Power',
            'give me more ',
            'the day forth ',
            'From',
        ]
        results = generate(model, bpe, texts, length=30)
        self.assertEqual(results[0][:len('Power, give me more power!')], 'Power, give me more power!')
