from glob import glob

from keras.models import load_model
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import parse_file


class TestGen:
    def __init__(self, batch_size=128):
        self.files = glob('data/test/audio/*.wav')
        self.batch_size = batch_size

    def get_data(self):
        while len(self.files):
            names, batch_sp, batch_mfcc = [], [], []
            while len(names) < self.batch_size and len(self.files):
                fname = self.files.pop()

                data, spectrogram, features = parse_file(fname)
                names.append(fname.split('/')[-1])
                batch_sp.append(spectrogram)
                batch_mfcc.append(features)

            batch_mfcc, batch_sp = map(np.array, (batch_mfcc, batch_sp))
            yield names, [batch_mfcc, batch_sp]

    def __len__(self):
        return np.ceil(len(self.files) / self.batch_size).astype(int)


def main():
    labels = 'yes no up down left right on off stop go silence unknown'.split(' ')
    models = glob(f'result/naive*.h5')
    models = [load_model(model) for model in models]

    gen = TestGen()

    probas = []
    result = []
    for names, batch in tqdm(gen.get_data(), desc='predicting', total=len(gen)):
        preds = np.zeros((len(names), 12))
        for i, model in enumerate(models):
            preds += model.predict(batch, batch_size=len(names))

        for name, prob in zip(names, preds):
            d = {labels[i]: prob[i] for i in range(len(labels))}
            d['fname'] = name
            probas.append(d)

        for p, n in zip(preds, names):
            label = labels[p.argmax()]
            result.append({'fname': n, 'label': label})

    df = pd.DataFrame(result)
    df.to_csv('result/submit.csv', index=False)

    probas = pd.DataFrame(probas)
    probas.to_csv('result/probas.csv', index=False)


if __name__ == '__main__':
    main()
