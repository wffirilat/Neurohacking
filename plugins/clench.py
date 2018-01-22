# -*- coding: utf-8 -*-
"""
Project: neurohacking
File: clench.py.py
Author: wffirilat
"""

import numpy as np
import time

import sys

import plugin_interface as plugintypes
from open_bci_v3 import OpenBCISample


class PluginClench(plugintypes.IPluginExtended):
    def __init__(self):
        self.release = True
        self.packetnum = -1
        self.threshold = None
        self.uthreshold = None
        self.ticknum = None
        self.storelength = 1024
        self.starttime = None
        self.state = 'unstarted'
        self.channel = 3
        self.restingmax, self.restingmin = 0, 0
        self.clenchmax, self.clenchmin = 0, 0
        self.unclenchmax, self.unclenchmin = 0, 0
        self.rawdata = np.zeros((8, self.storelength))
        self.data = np.zeros((8, self.storelength))

    def activate(self):
        print("clench activated")

    # called with each new sample
    def __call__(self, sample: OpenBCISample):
        if sample.id == 0:
            if self.packetnum == -1:
                self.starttime = time.time()
            self.packetnum += 1
        self.ticknum = self.packetnum * 256 + sample.id
        self.rawdata[:, (sample.id + 256 * self.packetnum) % self.storelength] = sample.channel_data
        self.data[:, (sample.id + 256 * self.packetnum) % self.storelength] = [v - avg for avg, v in zip(
            [sum(self.rawdata[i, :]) / self.storelength for i in range(8)],
            sample.channel_data
        )]

        if self.state != 'calibrated':
            self.calibratetick()
        else:
            self.tick()

    def calibratetick(self):
        # print(self.data)
        dt = time.time() - self.starttime
        if self.state == "unstarted":
            print("Prepare to calibrate")
            self.state = "positioning"
        elif self.state == "positioning":
            if dt > 4:
                print('Calibrating')
                self.state = 'resting'
        elif self.state == 'resting':
            if dt > 6:
                print("Resting data gathered; Prepare to clench")
                self.state = 'clench'
                return
            if self.current >= self.restingmax:
                self.restingmax = self.current
            if self.current <= self.restingmin:
                self.restingmin = self.current
        elif self.state == 'clench':
            if dt > 7:
                print("Clench NOW!")
                self.state = 'clenching'
                return
        elif self.state == 'clenching':
            if dt > 8:
                print('Unclench!!')
                self.state = 'postclench'
                return
            if self.current > self.clenchmax:
                self.clenchmax = self.current
            if self.current < self.clenchmin:
                self.clenchmin = self.current
        elif self.state == 'postclench':
            if dt > 9:
                self.threshold = self.restingmax + ((self.clenchmax - self.restingmax) / 2)
                if self.release:
                    self.uthreshold = self.restingmin + ((self.clenchmin - self.restingmin) / 2)
                self.state = 'calibrated'
                print(self.restingmax, self.restingmin, self.clenchmax, self.clenchmin)
                return
            if self.release:
                if self.current > self.unclenchmax:
                    self.unclenchmax = self.current
                if self.current < self.unclenchmin:
                    self.unclenchmin = self.current
    
    @property
    def current(self):
        return self.data[self.channel, self.ticknum % self.storelength]

    def tick(self):
        if self.current > self.threshold:
            print(f" {self.ticknum}: Clenched!!")
        if self.release:
            if self.current < self.uthreshold:
                print(f" {self.ticknum}: Clenched!!")

