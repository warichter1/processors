#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:37:51 2023

@author: wrichter
"""

import pandas as pd
# import numpy as np
# import datetime
# import time
import copy

from cpuConf import amdRename, amdDrop, intelRename, intelDrop
# sourceDir = 'cpudb'
# dateCol = ['date', 'hw_avail.spec_int95']

class Processors:
    def __init__(self):
        self.sourceDir = 'cpudb'
        self.dateCol = ['date', 'hw_avail.spec_int95']
        self.filter = {'q1': '1/1/', 'q2': '4/1/', '2q': '4/1/', 'q3': '9/1/', '3q': '9/1/', 'q4': '10/1/',
                        'september': '9/1/', '0416': '4/1/16'}
        self.normalize = [' Intel®', 'Intel® ', ' Processors', ' Processor', ' (AF)', '®', ' Series', ' Scalable',
                          ' Family', ' Product', '™']
        self.lookup = {}
        self.loadtables()

    def loadtables(self):
        names = ['processor_family', 'microarchitecture', 'manufacturer', 'technology', 'code_name', 'gate_delay', 'mips_est']

        for name in names:
            self.lookup[name] = self.loadFile(f'./{self.sourceDir}/{name}.csv').fillna(0)

    def datetime_to_epoch(self, ser):
        """Don't convert NaT to large negative values."""
        if ser.hasnans:
            res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
        else:
            res = ser.astype('int64')
        return res // 10**9

    def fixDate(self, cell):
        """Normalize the date format to allow conversion to EPOCH format."""
        if pd.isnull(cell):
            return cell
        if type(cell) == str:
            cell = cell.lower().replace(' ', '').replace("'", '')
        else:
            return cell
        if 'oem:' in cell:
            return cell.split('oem:')[1]

        for key in self.filter:
            if key in cell:
                return cell.replace(key, self.filter[key])
        return cell

    def loadFile(self, filename):
        pd.set_option('future.no_silent_downcasting', True)
        return pd.read_csv(filename)

    def getProcessFamilyId(self, row, vendor, label, fields=['hw_model']):
        """Process the rows in a processor file to convert to CPUDB format."""
        for field in fields:
            cell = row[field].replace('™', '')
            for key in self.normalize:
                cell = cell.replace(key, '')
            for modelId in vendor.index:
                model = vendor.loc[modelId]
                if type(row[label]) == str:
                    hwModel = row[label].replace('™', '')
                else:
                    hwModel = "NA"
                if model['name'].lower() in cell.lower():
                    return model['processor_family_id']
                if 'ryzen' in cell.lower() or 'epyc' in cell.lower():
                    if model['name'] in self.getRyzen(cell):
                        return model['processor_family_id']
                if model['name'] in hwModel:
                    return model['processor_family_id']
                if 'AMD' not in cell and model['name'] in self.getASeries(cell):
                    return model['processor_family_id']
        print('Not found: ', row[field], '-', cell)
        return None

    def getFamilyId(self, df, manufacturerId, family="./cpudb/processor_family.csv", label='hw_model.spec_int2k6'):
        """Add the proccessor family ID."""
        processorFamily = self.loadFile(family)
        vendorFamily = copy.copy(processorFamily.loc[processorFamily['manufacturer_id'] == manufacturerId])
        df['processor_family_id'] = [self.getProcessFamilyId(df.loc[rowId], vendorFamily, label) for rowId in df.index]
        return df

    def importIntel(self, filename, folder='Intel', manufacturerId=9):
        """Import and process Intel processor specs."""
        df = None
        with open(f"{folder}/{filename}", "r") as f:
            intelFiles = f.readlines()
        for file in intelFiles[0].replace('\n', '').split(','):
            print('Processing', file)
            if df is None:
                df = self.processIntelFile(file)
            else:
                df = pd.concat([df, self.processIntelFile(file, len(df))], join='outer', ignore_index=True)
        df = df.assign(test_sponsor='Intel')
        df['manufactuer'] = 'Intel'
        df.fillna(0, inplace=True)
        df = self.getFamilyId(df, manufacturerId, label='hw_model')
        df['date'] = [self.fixDate(cell) for cell in df['date']]
        return df

    def importAmd(self, filename, manufacturerId=1, family="./cpudb/processor_family.csv"):
        """Process AMD records."""
        df = self.loadFile(f'./{self.sourceDir}/{filename}').drop(columns=amdDrop)
        df.rename(columns=amdRename, inplace=True)
        df.assign(manufacturer_id=manufacturerId, inplace=True)
        df['processor_id'] = [4000 + i for i in range(len(df))]
        df['manufactuer'] = 'AMD'
        df = df.assign(test_sponsor='AMD')
        df.fillna(0, inplace=True)
        df['date'] = [self.fixDate(cell) for cell in df['date']]
        return self.getFamilyId(df, manufacturerId)

    def getRyzen(self, cell):
        """Find the submodel."""
        if 'Radeon' in cell:
            return 'Ryzen with Radeon Graphics'
        rowArr = cell.split(' ')
        name = rowArr[1]
        if 'threadripper' in cell.lower():
            name += ' Threadripper'
        if 'pro' in cell.lower():
            name += ' PRO'
        if len(rowArr) > 5:
            code = rowArr[3][0][0]
        else:
            code = rowArr[-1:][0][0]
        name += f' {code}000'
        return name

    def getASeries(self, cell):
        """Parse an AMD A Series Processor record."""
        rowArr = cell.split(' ')
        name = f"{rowArr[0].split('-')[0]}-Series"
        if 'a' == cell[0].lower() and 'radeon' in cell.lower():
            return f'{name} APU'
        else:
            return name

    def processIntelFile(self, filename, indexStart=0):
        """Read Intel Ark export files with records as columns, convert to rows."""
        IntelIndex = 6000 + indexStart
        df = pd.read_csv(f"./Intel/{filename}", skiprows=2)
        df = df.rename(columns={' ': 'model'}).set_index('model').T.rename_axis('model').reset_index(drop=True)
        df.rename(columns=intelRename, inplace=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.drop(columns=intelDrop, errors='ignore', inplace=True)
        df = df.dropna(how='all')
        df = df.dropna(how='all', axis=1)
        df['processor_id'] = [IntelIndex + i for i in range(len(df))]
        return df

    def loadBaseProcessors(self):
        """Load base processors."""
        # pd.set_option('future.no_silent_downcasting', True)
        processor = self.loadFile(f"./{self.sourceDir}/processor.csv")
        specint2k6 = self.loadFile(f"./{self.sourceDir}/spec_int2006.csv")
        specint2k0 = self.loadFile(f"./{self.sourceDir}/spec_int2000.csv")
        specint95 = self.loadFile(f"./{self.sourceDir}/spec_int1995.csv")
        specint92 = self.loadFile(f"./{self.sourceDir}/spec_int1992.csv")

        df = processor.merge(specint2k6, on="processor_id", suffixes=(".proc", ".spec_int2k6"), how='outer')
        df = df.merge(specint2k0, on="processor_id", how='outer',
                      suffixes=(".spec_int2k6", ".spec_int2k0"))
        df = df.merge(specint95, on="processor_id", how='outer',
                      suffixes=(".spec_int2k0", ".spec_int95"))
        df = df.merge(specint92, on="processor_id", how='outer',
                      suffixes=(".spec_int95", ".spec_int92"))
        for field in self.dateCol:
            df[field] = self.datetime_to_epoch(pd.to_datetime(pd.Series(df[field])))
        df["max_clock"] = df["max_clock"].fillna("clock")
        df.max_clock = df.clock.where(df.max_clock == 'clock', df.max_clock)
        df["perfnorm"] = df["basemean.spec_int2k6"] / df["tdp"]
        df = df.fillna(0)
        return df


if __name__ == "__main__":
    processors = Processors()
    specIntel2k23 = processors.importIntel('intel.txt')
    specAmd2k23 = processors.importAmd("AMDcpu.csv")
    baseDf = processors.loadBaseProcessors()
    # baseSpec = dict(spec=dict(s95to2k0=baseDf[['basemean.spec_int2k0', 'basemean.spec_int95']].mean(axis=1),
    #                           s2k0to2k6=baseDf[['basemean.spec_int2k6', 'basemean.spec_int2k0']].mean(axis=1),
    #                           no95=baseDf["basemean.spec_int95"].isna(), no2k0=baseDf["basemean.spec_int2k0"].isna(),
    #                           no2k6 = baseDf["basemean.spec_int2k6"].isna()),
    #                 scale=dict(scaleclk=min(baseDf["max_clock"].values), scaletrans=baseStats["transistors"]['min'],
    #                            scaletdp=baseStats["tdp"]['min'], scaleperf=baseStats["basemean.spec_int2k6"]['min'],
    #                            scaleperfnorm=baseStats["perfnorm"]['min']))
