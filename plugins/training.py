# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: training.py
Author: wffirilat
"""
import numpy as np

from numpy.fft import fft, rfft

import plugin_interface as plugintypes

from enum import Enum

class States(Enum):
    INIT = 'INIT'
    ACQUIRE = 'ACQUIRE'
    TRAIN = 'TRAIN'
    VALIDATE = 'VALIDATE'
    INTERACT = 'INTERACT'

actions = [
    'Right',
    'Left'
]

class PluginTraining(plugintypes.IPluginExtended):
    def __init__(self):
        self.packetnum = -1
        self.ticknum = -1
        self.storelength = 1024
        self.rawdata = np.zeros((8, self.storelength))
        self.data = np.zeros((8, self.storelength))
        self.state = States.INIT
        self.nn = ...  # TODO
        self.actiondata = {action: [] for action in actions}



    def activate(self):
        print("helpers activated")

    # called with each new sample
    def __call__(self, sample):
        self.ticknum += 1
        if sample.id == 0:
            self.packetnum += 1
        self.rawdata[:, (sample.id + 256 * self.packetnum) % self.storelength] = sample.channel_data
        self.data[:, (sample.id + 256 * self.packetnum) % self.storelength] = [v - avg for avg, v in zip(
            [sum(self.rawdata[i, :]) / self.storelength for i in range(8)],
            sample.channel_data
        )]
        if self.state is States.INIT and self.ticknum >= self.storelength:
            print('Get ready for **INSTRUCTIONS**')
            self.state = States.ACQUIRE

        ...

        if self.state is States.INTERACT:
            self.interact(sample)


    def process(self, data):
        ...  # TODO
        self.actiondata[self.nextinst] = ...

        print(self.getRandInstruction())
        self.nextinst = ... # ^that

    def train(self):
        """Trains incrementally (we think)"""
        ...  # TODO

    def validate(self): ...  # TODO
        # getData()



    def channelHistory(self, channel):
        d = list(self.data[channel, self.ticknum % self.storelength:])
        d.extend(self.data[channel, :self.ticknum % self.storelength])
        return d

    def getRandInstruction(self):
        pass

    def interact(self, sample):
        self.nn.get(sample)
        ...  # TODO
