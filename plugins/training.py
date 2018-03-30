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
from neuralnet import NeuralNet, Model
import random
import keyboard
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
        self.nn = NeuralNet(0.2, Model.LR, 'accuracy')
        self.actiondata = {action: [] for action in actions}
        self.recorded = np.zeros([8,0])
        self.pressed = False

    def activate(self):
        print("training activated")

    # called with each new sample
    def __call__(self, sample):
        if keyboard.is_pressed('q'): #True if q is pressed
            print("WHAT!")
            self.pressed = True
        '''else:
            self.pressed = False'''
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
        if self.state in (States.ACQUIRE, States.GATHERED, States.TRAIN):
            if self.packetnum > 10: #So this goes after 10 seconds which is good?
                '''When it has ten data points''' #Not actually.
                # self.nn.data(self.recorded)
                self.nn.train(self.recorded)
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
        #print("In process: State is ", self.state)
        if self.state == States.ACQUIRE:
            instruction = actions[random.randint(0, 1)]
            print("Move your hand %s" % (instruction))
            self.state = States.TRAIN
        if (self.state == States.TRAIN) & self.pressed: #Only evolves if q was pressed (problem is this happens multiple times a press)
                temp = data
                #Todo remember that this only takes the recent x number of packets
                self.state = States.GATHERED
                self.pressed = False

        if self.state == States.GATHERED:
            temp = fft(temp, 60) #Plot twis this is both real and imaginary
            #print(temp)
            print(temp.shape)
            self.recorded = np.hstack((self.recorded, temp))#Todo figure out what this does HINT IT STACKS THEM HORIZONTALLY
            print("self.recorded.shape =", self.recorded.shape) #Yeah I realize this is a real problem
            print(self.recorded) # So like every 8:60 is its own data set (Variable with fft size
            #This should be a nxm matrix i think
            self.state = States.ACQUIRE
            # Fft the data and store it with the instruction to recorded

    def train(self):
        """Trains incrementally (we think - not right now)"""
        ...  # TODO

    def validate(self):
        ...  # TODO

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
