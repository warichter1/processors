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


v7Details = 'ARM-7a_details.csv'
v8Details = 'ARM-8a_details.csv'
designs = 'ARM-designs.csv'
thrdPartyDesigns = 'ARM_third_party.csv'
timeline = 'ARM_timeline.csv'
vendorModel = 'ARM_vendor.csv'
vendorModelOther = 'ARM_vendor_model_other.csv'

pd.options.mode.chained_assignment = None


class ArmDataProcessor:
    """Parse wilk data for ARM processors into a standardized datafram format."""

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
            # print(index)
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
                # print(buf)
            else:
                soc = "0"
            vendor = item.split(':') if ':' in item else ['0','0']
            core.append({'family': family, 'vendor': vendor[0], 'model': [x.strip() for x in vendor[1].split(',')], 'soc_used': soc})
        return core

    def loadThirdParty(self):
        """Process Feature, Cache and mips columms into standard columns."""
        self.dfThird = pd.read_csv(f'{self.folder}/{self.thirdPartyDesigns}')  # .set_index('family')
        self.columns = {'cores': 'strip', 'hw_nthreadspercore': None, 'clock': 0, 'max_clock': 1, 'Wireless': 'strip',
                        'Wireless2': 'strip','decode': 'split', 'out_of_order': True, 'wide': 'split', 'issue': 'split',
                        'superscalar': True, 'AArch64': True, 'pipeline': 'strip', 'bus_width': 'split', 'virtualization': True,
                        'transistors': True, 'dynamic code optimization': True, 'optimization cache': 'strip',
                        'L0:': 'strip', 'L1:': 'strip', 'L2:': 'strip', 'SLC:': 'strip', 'die_size': 'strip',
                        'L1I:':'strip','L1D:': 'strip', 'DSP': 'strip', 'L3:': 'strip', 'SMP': True, 'Thumb': True,'Thumb-2': True,
                        'FPU': 'strip','TrustZone': True, 'NEON': True, 'SIMD': True, 'D-cache': 'strip', 'MMU': True}
        self.cleanColumns = {'cores': 'hw_ncores', 'L0:': 'l0_cache', 'L1:': 'l1_cache', 'L2:': 'l2_cache',
                             'L3:': 'l3_cache', 'SLC:': 'SLC', 'L1I:': 'L1I_cache', 'L1D:': 'L1D_cache',}
        self.labels = ['Feature', 'Cache (ID), MMU', 'mips']
        for col in list(self.columns):
            self.dfThird[col] = '0'
        for dfIndex in self.dfThird.index:
            for label in self.labels:
                match label:
                    case 'mips':
                        self.parseClock(label, dfIndex)
                    case _:
                        self.parseThirdParty(label, dfIndex)
        self.thirdCleanup()

    def thirdCleanup(self):
        """After values within self.labels field names are parsed, remove the original columns."""
        self.dfThird.rename(columns=self.cleanColumns, inplace=True)
        self.dfThird.drop(columns=self.labels, inplace=True)

    def parseClock(self, label, dfIndex):
        """Clock field (mips) is a different format, parse just this field. Can hold 1 or more entries."""
        clock = 0
        field = str(self.dfThird.loc[dfIndex][label]).strip().replace('\n', ',')
        field = unicodedata.normalize('NFKD', field).split(',')
        if clock == 0:
            self.dfThird.loc[dfIndex]['clock'] = field[clock].strip()
            clock += 1
        if len(field) > 1:
            self.dfThird.loc[dfIndex]['max_clock'] = field[clock].strip()


    def parseThirdParty(self, label, dfIndex):
        """A generalied parser for Feature and cache columns,"""
        for field in str(self.dfThird.loc[dfIndex][label]).strip().replace('\n', ',').split(','):
            field = field.strip()
            self.parseCell(field, dfIndex)

    def parseCell(self, field, dfIndex):
        """After first level parsing deal with edge cases first."""
        for key in list(self.columns):
            if key in field:
                value = self.columns[key]
                if type(value) == bool:
                    self.dfThird.loc[dfIndex][key] = True if field != 'No MMU' else False
                else:
                    result = getValue(field, value, key)
                    self.dfThird.loc[dfIndex][key] = result[0]


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
    test = arm.mergeVendor()
    arm.loadThirdParty()
    # df = arm.df['family'].merge(arm.df['timeline'], on="family", how='outer')
