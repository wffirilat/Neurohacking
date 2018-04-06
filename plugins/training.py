# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: training.py
Author: wffirilat
"""
import typing

import numpy as np

from numpy.fft import fft, rfft

import plugin_interface as plugintypes

from enum import Enum, auto
from neuralnet import NeuralNet, Model
import random
import keyboard
import time

class States(Enum):
    INIT = auto()
    ACQUIRE = auto()
    TRAIN = auto()
    VALIDATE = auto()
    INTERACT = auto()
    GATHERED = auto()

Action = typing.NewType("Action", str)

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
        self.fftsize = 38
        self.actiondata: typing.Dict[Action: np.ndarray] = {action: [] for action in actions}
        self.recorded = np.zeros([0,8*(int(self.fftsize/2)+1)+1])
        self.pressed = False
        self.instruction = "None"

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
            if self.packetnum > 10 and len(self.recorded)>0:
                #When 10 seconds have passed
                #self.nn.data(self.recorded)
                print(self.recorded.shape)

                self.nn.finalizeTraining(self.recorded) #Before this I probs need to attach the information about right or left in here at end
                self.nn.quality() #Todo This gives an issue with too many samples I think?
                self.state = States.INTERACT
            else:
                self.process(self.data)

        if self.state is States.INTERACT:
            self.interact(self.data)

    def process(self, data):
        if self.state == States.ACQUIRE:
            self.instruction = actions[random.randint(0, 1)]
            print("Move your hand %s" % (self.instruction))
            self.state = States.TRAIN
        if (self.state == States.TRAIN) & self.pressed: #Only evolves if q was pressed (problem is this happens multiple times a press)
                temp = data
                #Todo remember that this only takes the recent x number of packets
                self.state = States.GATHERED
                self.pressed = False

        if self.state == States.GATHERED:
            temp = rfft(temp, self.fftsize) #Plot twist this is both real and imaginary Consider using bayervillege

            #Todo also this gives issue when only taking real values (ask kindler)
            print("   ")
            print(temp.shape)
            temp = temp.flatten().real
            print(temp.shape)

            temp = np.append(temp, self.instruction)
            print(temp.shape)
            self.recorded = np.vstack((self.recorded, temp)) #Todo This bugs out if fft doesnt find enough pattern
            print("self.recorded.shape =", self.recorded.shape) #Yeah I realize this is a real problem
            self.state = States.ACQUIRE

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
