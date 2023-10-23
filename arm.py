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
        self.df = {}
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
        self.df['timeline'] = pd.read_csv(f'{self.folder}/{self.timeline}').set_index('family')
        df = pd.read_csv(f'{self.folder}/{filename}').set_index('family')
        df.index = df.index.str.strip()
        df.rename(columns={'soc': 'socImport', 'products': 'productsImport'}, inplace=True)
        print(f'Processing: {filename}')
        processedList = {}
        for col in self.vendor[filename]:
            df[f'{col}Import'] = df[f'{col}Import'].str.strip()
            df[f'{col}'] = 0
            df[f'{col}Buffer'] = 0
            processedList[col] = []
        df['vendor'] = 0
        df = df.fillna(0)
        for index in df.index:
            print(index)
            for col in self.vendor[filename]:
                processedList[col] += self.parseVendor(index, str(df.loc[index][f'{col}Import']).replace('\n', '').split(';'))
        self.df['family'] = pd.DataFrame(processedList['soc']).explode('model',
                                                                    ignore_index=False).rename(columns={'model': 'soc'}).drop('soc_used', axis=1).reset_index(drop=True)
        self.df['family'] = self.df['family'].merge(self.df['timeline'], on="family", how='outer')
        self.df['products'] = pd.DataFrame(processedList['products']).explode('model', ignore_index=False).reset_index(drop=True).add_prefix('product')
        return processedList

    def parseVendor(self, family, row):
        """Process a row of data and store the results."""
        """{'name': None, 'soc': None, 'product': []}}"""
        core = []
        if str(row[0]) == "0" or row[0].lower() == family.lower():
            return [{'family': family, 'vendor': '0', 'model': '0', 'soc_used': '0'}]
        for item in row:
            if '~' in item:
                buf = item.split('~')
                soc = buf[0]
                item = buf[1]
                print(buf)
            else:
                soc = "0"
            vendor = item.split(':') if ':' in item else ['0','0']
            # core.append({'family': family, 'vendor': vendor[0], 'model': vendor[1].split(','), 'soc_used': soc})
            core.append({'family': family, 'vendor': vendor[0], 'model': [x.strip() for x in vendor[1].split(',')], 'soc_used': soc})
        return core

    def loadThirdParty(self):
        df = pd.read_csv(f'{self.folder}/{self.thirdPartyDesigns}').set_index('family')
        for column in ['hw_ncores', 'hw_nthreadspercore', 'clock', 'max_clock', 'bus_width', 'transistors', 'l1_cache', 'l2_cache', 'die_size']:
            df.insert(2, column, 1 if column == 'hw_nthreadspercore' else 0, allow_duplicates=False)
        return df




if __name__ == "__main__":
    arm = ArmDataProcessor('ARM')
    test = arm.mergeVendor()
    # df = arm.df['family'].merge(arm.df['timeline'], on="family", how='outer')
