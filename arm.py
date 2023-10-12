#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:48:46 2023

@author: wrichter

ARM processor builder script
"""
import pandas as pd
import numpy as np
import datetime
import time
import copy


v7Details = 'ARM-7a_details.csv'
v8Details = 'ARM-8a_details.csv'
designs = 'ARM-designs.csv'
thrdPartyDesigns = 'ARM_third_party.csv'
timeline = 'ARM_timeline.csv'
vendorModel = 'ARM_vendor.csv'
vendorModelOther = 'ARM_vendor_model_other.csv'

pd.options.mode.chained_assignment = None

class ArmDataProcessor:
    def __init__(self, dataFolder):
        self.df = None
        self.folder = dataFolder
        self.v7Details = 'ARM-7a_details.csv'
        self.v8Details = 'ARM-8a_details.csv'
        self.designs = 'ARM-designs.csv'
        self.thirdPartyDesigns = 'ARM_third_party.csv'
        self.timeline = 'ARM_timeline.csv'
        self.vendor = {'ARM_vendor.csv': ['soc', 'products']}

    def mergeVendor(self):
        """Normalize wikipedia formatted vendor table."""
        filename = list(self.vendor.keys())[0]
        self.df = pd.read_csv(f'{self.folder}/{filename}').fillna(0)
        print(f'Processing: {filename}')
        for col in self.vendor[filename]:
            self.df[f'{col}Buffer'] = 0
        for index in self.df.index:
            family = str(self.df.loc[index]['family']).replace('\n', '').replace('\xa0', '').strip(' ')
            print(family)
            for col in self.vendor[filename]:
                self.df.at[index, f'{col}Buffer'] = str(self.df.loc[index][col]).replace('\n', '').split(';')


    def parseRow(self, row):
        """Process a row of data and store the results."""
        details = {'vendor': {'name': None, 'soc': None, 'product': []}}
        if row[0] == 0:
            return row
        for item in row:
            if '~' in item:
                core = item.split('~')


if __name__ == "__main__":
    arm = ArmDataProcessor('ARM')
    arm.mergeVendor()
    # vendorDf, record = mergeVendor({vendorModel: 'soc', vendorModelOther: 'products'})
    # vendorDf, record = mergeVendor({vendorModel: ['soc','products']})
