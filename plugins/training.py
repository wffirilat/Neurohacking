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
from neuralnet import NeuralNet
import random
import time


class States(Enum):
    INIT = 'INIT'
    ACQUIRE = 'ACQUIRE'
    TRAIN = 'TRAIN'
    VALIDATE = 'VALIDATE'
    INTERACT = 'INTERACT'
    GATHERED = 'GATHERED'

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
        self.nn = NeuralNet(0.2, 'LR','accuracy')
        self.actiondata = {action: [] for action in actions}
        self.recorded = np.zeros((1000,))
        self.adder = 0



    def activate(self):
        print("training activated")

    # called with each new sample
    def __call__(self, sample):
        self.ticknum += 1
        if sample.id == 0:
            if self.packetnum == -1:
                self.starttime = time.time()
            self.packetnum += 1
        self.rawdata[:, (sample.id + 256 * self.packetnum) % self.storelength] = sample.channel_data
        self.data[:, (sample.id + 256 * self.packetnum) % self.storelength] = [v - avg for avg, v in zip(
            [sum(self.rawdata[i, :]) / self.storelength for i in range(8)],
            sample.channel_data
        )]
        if self.state is States.INIT and self.ticknum >= self.storelength:
            print('Get ready for **INSTRUCTIONS**')
            self.state = States.ACQUIRE
        if self.state is States.ACQUIRE:
            if self.adder>10:
                '''When it has ten data points'''
                self.nn.data(self.recorded)
                self.nn.train()
                self.nn.quality()
                self.state = States.INTERACT
            else:
                self.process(self.data)

        if self.state is States.INTERACT:
            self.interact(self.data)


    def process(self, data):
        '''        ...  # TODO IDK what any of this was supposed to do
        self.actiondata[self.nextinst] = ...

        print(self.getRandInstruction())
        self.nextinst = ... # ^that
        '''
        #Do this step only once
        dt = time.time()-self.starttime()
        if self.state == States.ACQUIRE:
            instruction = actions[random.randint(0,1)]
            print("Move your hand %s")%(instruction)
            self.state == States.TRAIN
        if self.state == States.TRAIN:
            if dt>1 and dt<2:
                temp = data #Yeah I get that this is bad but what can you do?
                self.state = States.GATHERED

        if self.state == States.GATHERED:
            temp = fft(temp)
            print(temp)
            self.recorded = np.append(self.recorded, temp)
            self.adder += 1
            self.state = States.ACQUIRE
        #Fft the data and store it with the instruction to recorded


    def train(self):
        """Trains incrementally (we think - not right now)"""
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
        sample = fft(sample)
        print(self.nn.get(sample))

