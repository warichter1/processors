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
import unicodedata
from copy import copy

pd.options.mode.chained_assignment = None

armSelect = ['family', 'vendor', 'soc',	'core_type', 'architecture_x', 'Year', 'productvendor', 'productmodel', 'productsoc_used', 'hw_model']

def armLoader(folder, key='merged', fileTemplate='arm', columns=None):
    source = f'{folder}/{fileTemplate}_{key}.csv'
    return pd.read_csv(source) if columns is None else pd.read_csv(source, usecols=columns)

class ArmDataProcessor:
    """Parse wilk data for ARM processors into a standardized datafram format."""

    def __init__(self, dataFolder):
        self.designColumns = {'cores': 'strip', 'hw_nthreadspercore': None, 'clock': 0, 'max_clock': 1,
                              'Wireless': 'strip', 'Wireless2': 'strip', 'decode': 'split', 'out_of_order': True,
                              'wide': 'split', 'issue': 'split', 'superscalar': True, 'AArch64': True,
                              'addressing': 'strip', 'memory': 'strip', 'system timer': 'strip', 'Saturated': True,
                              'branch prediction': 'strip', 'instruction fetch': 'strip', 'AArch32': True,
                              'pipeline': 'strip', 'bus_width': 'split', 'virtualization': True, 'transistors': True,
                              'dynamic code optimization': True, 'optimization cache': 'strip', 'L0:': 'strip',
                              'L1:': 'strip', 'L2:': 'strip', 'SLC:': 'strip', 'die_size': 'strip', 'L1I:': 'strip',
                              'L1D:': 'strip', 'DSP': 'strip', 'L3:': 'strip', 'SMP': 'strip', 'Thumb': True,
                              'Thumb-1': True, 'Thumb-2': True, 'coprocessor bus': False, 'floating-point unit': True,
                              'profile': 'strip', 'DBX': 'strip', 'Co-processor': True, 'FPU': 'strip',
                              'instruction': 'strip', 'Divide': True, 'FPU': 'strip', 'TrustZone': True, 'NEON': True,
                              'SIMD': True, 'D-cache': 'strip', 'MMU': True, 'notes': None, 'cache': True, 'VFP': True,
                              'ACP': True, 'GIC': True, 'SCU': True, 'GIC': True, 'VFPv3': 'split', 'VFPv4': True,
                              'VFPv5': 'split', 'LLPP': True, 'TCM': True}
        self.cleanColumns = {'cores': 'hw_ncores', 'L0:': 'l0_cache', 'L1:': 'l1_cache', 'L2:': 'l2_cache',
                             'L3:': 'l3_cache', 'SLC:': 'SLC', 'L1I:': 'L1I_cache', 'L1D:': 'L1D_cache', 'Year ': 'Year'}
        self.df = {}
        self.dfDesign = {}
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
            # print(index)
            for col in self.vendor[filename]:
                processedList[col] += self.parseVendor(index, str(df.loc[index][f'{col}Import']).replace('\n', '').split(';'))
        self.df['family'] = pd.DataFrame(processedList['soc']).explode('model',
                                                                    ignore_index=False).rename(columns={'model': 'soc'}).drop('soc_used', axis=1).reset_index(drop=True)
        self.df['family'] = self.df['family'].merge(self.df['timeline'], on="family", how='outer')
        self.df['family'] = self.df['family'].fillna(0)
        self.df['products'] = pd.DataFrame(processedList['products']).explode('model', ignore_index=False).reset_index(drop=True).add_prefix('product')
        self.df['products'].fillna(0)
        # return processedList

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
            else:
                soc = "0"
            vendor = item.split(':') if ':' in item else ['0','0']
            core.append({'family': family, 'vendor': vendor[0], 'model': [x.strip() for x in vendor[1].split(',')],
                         'soc_used': soc})
        return core

    def loadArmDetails(self):
        """Import 7/8 Details."""
        df = pd.read_csv(f'{self.folder}/{self.v7Details}')
        df = pd.concat([df, pd.read_csv(f'{self.folder}/{self.v8Details}')], join='outer', ignore_index=True)
        self.df['details'] = df.fillna(0)


    def loadArmDesign(self, key='arm'):
        """Process design files."""
        self.dfDesign['arm'] = pd.read_csv(f'{self.folder}/{self.designs}')
        labels = ['Features', 'mips']
        self.processDesign(key, labels)
        self.loadThirdParty()
        df1 = copy(self.dfDesign['arm'].rename({'family ': 'family'}, axis=1))
        df1['manufacturer'] = 'arm'
        df1.rename(columns={'version': 'hw_model'}, inplace=True)
        self.df['designs'] = pd.concat([df1, copy(self.dfDesign['third'])], join='outer', ignore_index=True)
        self.df['designs'] = self.df['designs'].fillna(0)

    def loadThirdParty(self, key='third'):
        """Process Feature, Cache and mips columms into standard columns."""
        self.dfDesign['third'] = pd.read_csv(f'{self.folder}/{self.thirdPartyDesigns}')  # .set_index('family')
        labels = ['Feature', 'Cache (ID), MMU', 'mips']
        self.processDesign(key, labels)

    def processDesign(self, key, labels):
        """Parse designs."""
        print("Process Designs:", key, labels)
        self.labels = labels
        for col in list(self.designColumns):
            self.dfDesign[key][col] = '0'
        for dfIndex in self.dfDesign[key].index:
            for label in self.labels:
                match label:
                    case 'mips':
                        self.parseClock(key, label, dfIndex)
                    case _:
                        self.parseDesign(key, label, dfIndex)
        self.designCleanup(key)

    def designCleanup(self, key):
        """After values within self.labels field names are parsed, remove the original columns."""
        self.dfDesign[key]['source'] = key
        self.dfDesign[key].rename(columns=self.cleanColumns, inplace=True)
        self.dfDesign[key].drop(columns=self.labels, inplace=True)
        self.dfDesign[key] = self.dfDesign[key].fillna(0)

    def parseClock(self, key, label, dfIndex):
        """Clock field (mips) is a different format, parse just this field. Can hold 1 or more entries."""
        clock = 0
        field = str(self.dfDesign[key].loc[dfIndex][label]).strip().replace('\n', ',')
        field = unicodedata.normalize('NFKD', field).split(',')
        if clock == 0:
            self.dfDesign[key].loc[dfIndex]['clock'] = field[clock].strip()
            clock += 1
        if len(field) > 1:
            self.dfDesign[key].loc[dfIndex]['max_clock'] = field[clock].strip()

    def parseDesign(self, key, label, dfIndex):
        """Generalied parser for Feature and cache columns."""
        for field in str(self.dfDesign[key].loc[dfIndex][label]).strip().replace('\n', ',').split(','):
            field = field.strip()
            self.parseCell(key, field, dfIndex)

    def parseCell(self, design, field, dfIndex):
        """After first level parsing deal with edge cases first."""
        for key in list(self.designColumns):
            if key in field:
                value = self.designColumns[key]
                if type(value) == bool:
                    self.dfDesign[design].loc[dfIndex][key] = True if field != 'No MMU' else False
                else:
                    result = getValue(field, value, key)
                    self.dfDesign[design].loc[dfIndex][key] = result[0]

    def process(self, folder='ARM', fileTemplate='arm'):
        key = 'merged'
        self.mergeVendor()
        self.loadArmDesign()
        self.loadArmDetails()
        self.mergeTables()
        export = f'{folder}/{fileTemplate}_{key}.csv'
        print(f'Writing: {export}')
        self.df[key].to_csv(export, index=False)

    def mergeTables(self):
        df1 = copy(self.df['family'])
        df1.drop('Year ', axis=1, inplace=True)
        df1 = df1.merge(copy(self.df['timeline'].drop(['core_type', 'architecture'], axis=1)), how='left', left_on='family', right_on='family')
        df1.rename({'Year ': 'Year'})
        df1 = df1.merge(copy(self.df['products']), how='left', left_on='family', right_on='productfamily').drop('productfamily', axis=1)
        df1 = df1.merge(copy(self.df['designs']), how='left', left_on='family', right_on='family')
        self.df['merged'] = df1.merge(copy(self.df['details']), how='left', left_on='hw_model', right_on='hw_model')
        self.df['merged'].rename(columns=self.cleanColumns, inplace=True)


def getValue(field, value, key):
    """Using case types strip, split or _ (default), determine how to extract each value from a column."""
    match value:
        case 'strip':
            return [field.replace(key, '').replace('-', '').strip()]
        case 'split':
            return [field.split('-')[0].strip()]
        case _:
            return ['0']


if __name__ == "__main__":
    arm = ArmDataProcessor('ARM')
    arm.process()
