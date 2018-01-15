import os
from glob import glob
from hashlib import md5

import numpy as np
from tqdm import tqdm
from fire import Fire
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from src.utils import parse_file
from src.model import naivenet


def get_callbacks(model_name, fold):
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=6)
    checkpoint = ModelCheckpoint(f'result/{model_name}_{fold}.h5',
                                 monitor='val_loss',
                                 save_best_only=True, verbose=0)
    callbacks = [es, reducer, checkpoint]
    return callbacks


class Dataset:
    def __init__(self, batch_size=128, folds=(0, 1, 2, 3), subsample=True):
        self.labels = 'yes no up down left right on off stop go silence unknown'.split(' ')
        self.labels_idx = {w: i for i, w in enumerate(self.labels)}

        self.batch_size = batch_size
        self.folds = folds
        self.subsample = subsample

        self.x_data_mfcc, self.x_data_sp, self.y_data = self.read_data()

    @staticmethod
    def assign_fold(s):
        s = s.split('/')[-1].split('_')[0]
        return int(md5(s.encode()).hexdigest(), 16) % 5

    def read_data(self):
        train_path = 'data/train/audio'
        labels = sorted([x for x in os.listdir(train_path) if not x.endswith('_')])
        x_data_mfcc, x_data_sp, y_data = [], [], []

        for label in tqdm(labels, desc=f'{self.folds}'):
            filelist = glob(os.path.join(train_path, label) + '/*')
            filelist = (f for f in filelist if self.assign_fold(f) in self.folds)

            idx = self.labels_idx.get(label, self.labels_idx['unknown'])

            for i, (r, s, f) in enumerate([parse_file(x) for x in filelist]):
                if self.subsample:
                    if idx == self.labels_idx['unknown'] and np.random.rand() > .05:
                        continue
                x_data_mfcc.append(f)
                x_data_sp.append(s)
                y_data.append(idx)

        return map(np.array, (x_data_mfcc, x_data_sp, y_data))

    def __next__(self):
        idx = np.random.randint(0, self.x_data_mfcc.shape[0], self.batch_size)
        x_mfcc = self.x_data_mfcc[idx]
        x_sp = self.x_data_sp[idx]
        y = self.y_data[idx]

        return [x_mfcc, x_sp], to_categorical(y, len(self.labels))


def main(n_fold, subsample=1):
    subsample = bool(subsample)
    folds = np.arange(5)
    train = Dataset(folds=tuple(folds), subsample=subsample)
    val = Dataset(folds=tuple([n_fold]), subsample=subsample)

    model = naivenet(shape1=(99, 13), shape2=(129, 71), output_dim=len(train.labels))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    weights = np.ones(len(train.labels))
    if not subsample:
        weights[-1] = .05

    subsample = 'subsample' if subsample else 'full'
    model.fit_generator(train,
                        steps_per_epoch=500,
                        epochs=100,
                        callbacks=get_callbacks(f'naive_{subsample}', n_fold),
                        validation_data=val,
                        validation_steps=100,
                        class_weight=weights)


if __name__ == '__main__':
    Fire(main)
