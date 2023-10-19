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
        self.df = pd.read_csv(f'{self.folder}/{filename}').set_index('family')
        self.df.index = self.df.index.str.strip()
        self.df.rename(columns={'soc': 'socImport', 'products': 'productsImport'}, inplace=True)
        print(f'Processing: {filename}')
        processedList = {}
        for col in self.vendor[filename]:
            self.df[f'{col}Import'] = self.df[f'{col}Import'].str.strip()
            self.df[f'{col}'] = 0
            self.df[f'{col}Buffer'] = 0
            processedList[col] = []
        self.df['vendor'] = 0
        self.df = self.df.fillna(0)
        for index in self.df.index:
            print(index)
            # results = {}
            for col in self.vendor[filename]:
                # processedList[col] = []
                # self.df.at[index, f'{col}Buffer'] = str(self.df.loc[index][f'{col}Import']).replace('\n', '').split(';')
                # results[index][col] = self.parseRow(index, f'{col}Buffer')
                # results[index][col] = self.parseRow(index, str(self.df.loc[index][f'{col}Import']).replace('\n', '').split(';'))
                processedList[col] += self.parseRow(index, str(self.df.loc[index][f'{col}Import']).replace('\n', '').split(';'))
                # processedList[col].append(self.parseRow(index, str(self.df.loc[index][f'{col}Import']).replace('\n', '').split(';')))
            # processedList.append(results)
        return processedList

    # def parseRow(self, family, key):
    def parseRow(self, family, row):
        """Process a row of data and store the results."""
        """{'name': None, 'soc': None, 'product': []}}"""
        # core = {'vendor': {}}
        core = []
        # row = self.df.loc[family, key]
        # keyValue = key.replace('Buffer', '')
        if str(row[0]) == "0" or row[0].lower() == family.lower():
            return row[0]
        for item in row:
            if '~' in item:
                buf = item.split('~')
                soc = buf[0]
                item = buf[1]
                print(buf)
            else:
                soc = "0"
            vendor = item.split(':') if ':' in item else [0,0]
                # vendor = item.split(':')
                # print(item, vendor)
                # core['vendor'][vendor[0]] = {'model': vendor[1].split(','), 'soc_used': soc}
            core.append({'family': family, 'vendor': vendor[0], 'model': str(vendor[1]).split(','), 'soc_used': soc})
        return core


if __name__ == "__main__":
    arm = ArmDataProcessor('ARM')
    test = arm.mergeVendor()
    df = pd.DataFrame(test)
    # test = arm.parseRow('ARM7TDMI(-S)', 'socBuffer')
    # vendorDf, record = mergeVendor({vendorModel: 'soc', vendorModelOther: 'products'})
    # vendorDf, record = mergeVendor({vendorModel: ['soc','products']})
