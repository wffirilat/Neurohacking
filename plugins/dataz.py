# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: dataz.py
Author: wffirilat
"""

import plugin_interface as plugintypes
from open_bci_v3 import OpenBCISample
import numpy as np
import csv

class PluginDataz(plugintypes.IPluginExtended):
    def __init__(self):
        self.packetnum = -1
        self.storelength = 1024
        self.rawdata = np.zeros((8, self.storelength))
        self.data = np.zeros((8, self.storelength))

    def activate(self):
        print("DATAZ activated")

    # called with each new sample
    def __call__(self, sample: OpenBCISample):
        if sample.id == 0:
            self.packetnum += 1
        self.rawdata[:, (sample.id + 256 * self.packetnum) % self.storelength] = sample.channel_data
        self.data[:, (sample.id + 256 * self.packetnum) % self.storelength] = [v-avg for avg, v in zip(
                [sum(self.rawdata[i, :])/self.storelength for i in range(8)],
                sample.channel_data
            )]
        if self.data[3, (sample.id + 256 * self.packetnum) % self.storelength] > 350:
            print((sample.id + 256 * self.packetnum))

        # print(min(self.data[3, :]), max(self.data[3, :]))