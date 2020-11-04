import csv
import time

import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from keras.callbacks import TensorBoard


class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """

    def __init__(self, display, out_path, model_nr, initial_epoch, lr):
        self.model_nr = model_nr
        self.lr = lr
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self.epochnr = initial_epoch
        self.out_path = out_path
        self.step_info_filename = self.out_path + self.model_nr + '_steplossesinfo.txt'
        open(self.step_info_filename, 'w').close()
        self.switch = False
        self.start = time.time()
        self.end = time.time()
        self.startbig = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epochnr += 1

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            # print("finished batch")

            if self.switch == True:
                # print("finished batch1")
                self.end = time.time()
                # print(self.end - self.start)
                self.switch = False
            else:
                # print("finished batch2")
                self.start = time.time()
                self.switch = True

            if self.step == 50:
                print("50: ", time.time() - self.startbig)
            if self.step == 100:
                print("100: ", time.time() - self.startbig)
            if self.step == 150:
                print("150: ", time.time() - self.startbig)
            if self.step == 200:
                print("200: ", time.time() - self.startbig)

            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' %.4f' % (val)
                else:
                    metrics_log += ' %.4e' % (val)

            current_lr = K.get_value(self.model.optimizer.lr)
            with open(self.step_info_filename, 'a') as self.step_info_file:
                self.step_info_file.write('{} {} {} {} {} \n'.format(self.step,
                                                                     self.params['steps'], self.epochnr,
                                                                     metrics_log, current_lr))
            self.metric_cache.clear()


class CSV_NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """

    def __init__(self, display, out_path, model_nr, initial_epoch, separator=',', append=True):
        # naming of output file
        self.model_nr = model_nr
        self.out_path = out_path
        self.filename = self.out_path + self.model_nr + '_steplossesinfoCSV.csv'

        self.display = display  # saving frequency in steps
        self.epochnr = initial_epoch
        self.append = append

        self.sep = separator

        self.file_flags = ''
        self._open_args = {'newline': '\n'}

        self.step = 0
        self.metric_cache = {}

        open(self.filename, 'w').close()

    def on_epoch_end(self, epoch, logs=None):
        self.epochnr += 1

    def on_batch_end(self, batch, logs={}):
        self.step += 1

        if self.step == 1:
            keys = [i for i in logs.keys()]
            keys.append('lr')
            with open(self.filename, 'w', newline='') as f:
                write_outfile = csv.writer(f)
                write_outfile.writerow(keys)

        values = [i for i in logs.values()]
        values.append(K.get_value(self.model.optimizer.lr))
        with open(self.filename, 'a', newline='') as f:
            write_outfile = csv.writer(f)
            write_outfile.writerow(values)

        # restart accumulating losses
        self.metric_cache.clear()


class TensorBoardWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback."""

    def __init__(self, batch_gen, nb_steps, input_dim, batch_nr, learning_phase, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps  # Number of times to call next() on the generator.
        self.input_dim = input_dim
        self.batch_nr = batch_nr
        self.learning_phase = learning_phase

    def on_epoch_end(self, epoch, logs):

        for i, data in enumerate(self.batch_gen):
            (ib, tb) = data

            if self.learning_phase:
                self.validation_data = [ib[0], ib[1], tb[0], np.ones((self.batch_nr,)), False]
            else:
                self.validation_data = [ib[0], ib[1], tb[0], np.ones((self.batch_nr,))]

        return super().on_epoch_end(epoch, logs)
