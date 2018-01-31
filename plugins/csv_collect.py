import csv
import timeit
import numpy as np
import datetime

import plugin_interface as plugintypes

class PluginCSVCollect(plugintypes.IPluginExtended):
    def __init__(self, file_name="collect.csv", delim=",", verbose=False):
        super().__init__()
        self.packetnum = -1
        self.ticknum = -1
        self.storelength = 1024
        self.rawdata = np.zeros((8, self.storelength))
        self.data = np.zeros((8, self.storelength))
        now = datetime.datetime.now()
        self.time_stamp = '%d-%d-%d_%d-%d-%d' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
        self.file_name = self.time_stamp
        self.start_time = timeit.default_timer()
        self.delim = delim
        self.verbose = verbose

    def activate(self):
        if len(self.args) > 0:
            if 'no_time' in self.args:
                self.file_name = self.args[0]
            else:
                self.file_name = self.args[0] + '_' + self.file_name
            if 'verbose' in self.args:
                self.verbose = True

        self.file_name = self.file_name + '.csv'
        print("Will export CSV to:" + self.file_name)
        # Open in append mode
        with open(self.file_name, 'a') as f:
            f.write('%' + self.time_stamp + '\n')

    def deactivate(self):
        print("Closing, CSV saved to:" + self.file_name)
        return

    def show_help(self):
        print("Optional argument: [filename] (default: collect.csv)")

    @property
    def current(self):
        return self.data[:, self.ticknum % self.storelength]

    def __call__(self, sample):
        self.ticknum += 1
        if sample.id == 0:
            self.packetnum += 1
        self.rawdata[:, (sample.id + 256 * self.packetnum) % self.storelength] = sample.channel_data
        self.data[:, (sample.id + 256 * self.packetnum) % self.storelength] = [v - avg for avg, v in zip(
            [sum(self.rawdata[i, :]) / self.storelength for i in range(8)],
            sample.channel_data
        )]

        t = timeit.default_timer() - self.start_time

        # print(timeSinceStart|Sample Id)
        if self.verbose:
            print("CSV: %f | %d" % (t, sample.id))

        row = ''
        row += str(t)
        row += self.delim
        row += str(sample.id)
        row += self.delim
        for i in self.current:
            row += str(i)
            row += self.delim
        # remove last comma
        row += '\n'
        with open(self.file_name, 'a') as f:
            f.write(row)
