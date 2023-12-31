#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:48:46 2023

@author: wrichter

Nvidia processor builder script
"""

import pandas as pd
import numpy as np
from copy import copy
import csv
from datetime import datetime

importNvidia = True
num = 0
months = {'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April', 'May': 'May', 'Jun': 'June',
          'Jul': 'July', 'Aug': 'August', 'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'}


def nvidiaLoader(folder, key='full', fileTemplate='nvidia', columns=None):
    return pd.read_csv(f'{folder}/{fileTemplate}_{key}.csv')


def nvidiaHeader(folder, key='full', fileTemplate='nvidia'):
    with open(f'{folder}/{fileTemplate}_{key}.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            header = row
            break
    return header

class NvidiaImport:
    """Export Nvidia data from wikipedia and import into a pandas dataframe."""

    def __init__(self, folder, uri):
        self.cleanColumns = {'Model': 'hw_model', 'model': 'hw_model', 'Core clock (MHz)': 'clock', 'Businterface': 'Bus Interface',
                             'Model(Architecture)': 'architecture', 'Bus interface': 'Bus Interface',
                             'Archi-tecture': 'architecture', 'Archi- tecture': 'architecture', 'Model Units': 'hw_model',
                             'Micro-architecture Unnamed: 1_level_2': 'architecture',
                             'Launch Unnamed: 2_level_2': 'Launch', 'Chips Unnamed: 3_level_2': 'Chips',
                             'Core clock(MHz) Unnamed: 4_level_2': 'clock',
                             'Shaders Cuda cores(total) Unnamed: 5_level_2': 'shaders_cuda_cores',
                             'Shaders Base clock (MHz)': 'shaders_clock', 'Die\xa0size(mm2)': 'Die size (mm2)',
                             'Shaders Max boostclock (MHz)': 'shaders_max_clock', 'Core\xa0config':'Core config',
                             'Memory Size (MB)': 'memory_size',
                             'Memory Bandwidth (GB/s)': 'memory_bandwidth',
                             'Memory Bus type': 'memory_bus',
                             'Memory Bus width (bit)': 'memory_bus_width',
                             'Memory Bus type Unnamed: 8_level_2': 'memory_bus',
                             'Memory Bus width(bit) Unnamed: 9_level_2': 'memory_bus_width',
                             'Memory Size(GB) Unnamed: 10_level_2': 'memory_size',
                             'Memory Clock(MT/s) Unnamed: 11_level_2': 'memory_clock', 'Memory clock (MHz)': 'memory_clock',
                             'Memory Bandwidth(GB/s) Unnamed: 12_level_2': 'memory_bandwidth',
                             'Processing power (GFLOPS)2': 'processing power (GFLOPS)',
                             'Processing power (GFLOPS)': 'processing power (GFLOPS)',
                             'Processing power (GFLOPS)3 Half precision': 'processing power (GFLOPS)_half_precision',
                             'Processing power (GFLOPS) Double precision': 'processing power (GFLOPS)_double_precision',
                             'Processing power (GFLOPS) Half Precision': 'processing power (GFLOPS)_half_precision',
                             'Processing power (GFLOPS) Half precision': 'processing power (GFLOPS)_half_precision',
                             'Processing power (GFLOPS) HalfPrecision': 'processing power (GFLOPS)_half_precision',
                             'Processing power (GFLOPS) Single precision': 'processing power (GFLOPS)_single_precision',
                             'Processing power (GFLOPS) Single precision (Boost)': 'processing power (GFLOPS)_single_precision (boost)',
                             'Processing power (GFLOPS)3 Ray Tracing Performance': 'processing power (GFLOPS) Ray Tracing Performance',
                             'Processing power (GFLOPS)3 Single precision': 'processing power (GFLOPS)_single_precision',
                             'Processing power (GFLOPS) Half precisionTensor Core FP32 Accumulate Unnamed: 13_level_2': 'processing power (GFLOPS)_half_precision_tensor',
                             'Processing power (GFLOPS) Single precision(MAD or FMA) Unnamed: 14_level_2': 'processing power (GFLOPS)_single_precision',
                             'Processing power (GFLOPS) Double precision(FMA) Unnamed: 15_level_2': 'processing power (GFLOPS)_double_precision',
                             'Processing power (TFLOPS) Tensor compute (FP16) (2:1\xa0sparse)': 'processing power (TFLOPS) Tensor compute (FP16) (2:1 sparse)',
                             'Processing power (GFLOPS) Tensor': 'processing power (GFLOPS) Tensor',
                             'Processing power (GFLOPS) Tensor compute (FP16)': 'processing power (GFLOPS) Tensor compute (FP16)',
                             'Processing power (GFLOPS) Tensor compute (FP16) (sparse)': 'processing power (GFLOPS) Tensor compute (FP16) (sparse)',
                             'Processing power (GFLOPS) Tensor compute + Single precision': 'processing power (GFLOPS) Tensor compute + Single precision',
                             'Processing power (TFLOPS) Tensor compute (FP16)': 'processing power (TFLOPS) Tensor compute (FP16)',
                             'Processing power (TFLOPS) Tensor compute (FP16) (2:1 sparse)': 'processing power (TFLOPS) Tensor compute (FP16) (2:1 sparse)',
                             'Processing power (GFLOPS) Tensor compute (FP16) (sparse)': 'processing power (GFLOPS) Tensor compute (FP16) (sparse)',
                             'Processing\xa0power\xa0(GFLOPS) Doubleprecision': 'processing power (GFLOPS)_double_precision',
                             'Processing\xa0power\xa0(GFLOPS) Halfprecision': 'processing power (GFLOPS)_half_precision',
                             'Processing\xa0power\xa0(GFLOPS) Singleprecision': 'processing power (GFLOPS)_single_precision',
                             'Processing\xa0power\xa0(GFLOPS) Tensorcompute(FP16)': 'processing power (TFLOPS) Tensor compute (FP16)',
                             'Processing power (TFLOPS) Double precision': 'processing power (TFLOPS)_double_precision',
                             'Processing power (TFLOPS) Half precision': 'processing power (TFLOPS)_half_precision',
                             'Processing power (TFLOPS) Single precision': 'processing power (TFLOPS)_single_precision',
                             'Ray\xa0tracing Performance RTX\xa0OPS(Trillions)': 'Ray-tracing Performance RTX OPS/s (Trillions)',
                             'CUDAcomputecapability Unnamed: 16_level_2': 'cuda_compute_capability',
                             'Transistors (million)Die size (mm2)': 'Transistors (million)',
                             'Transistors(billion)': 'Transistors (billion)', 'TDP (watts)': 'tdp', 'TDP(Watts)': 'tdp',
                             'TDP(watts) W': 'tdp', 'TDP(watts)': 'tdp', 'form factor Unnamed: 18_level_2': 'form_factor',
                             'GeForce (List of GPUs)': 'Features', 'GeForce (List of GPUs).1': 'gpu',
                             'Other products': 'Features', 'Other products.1': 'other_products',
                             'Software and technologies': 'Features', 'Features nFiniteFX II Engine': 'Features',
                             'Software and technologies.1': 'software_technologies',
                             'vteGraphics processing unit': 'category',
                             'vteGraphics processing unit.1': 'Features'}
        self.cleanDrop = ['Unnamed: 4_level_0 Unnamed: 4_level_1', 'Unnamed: 5_level_0 Unnamed: 5_level_1']
        self.fullColumns = {'hw_model_x': 'hw_model', 'Launch_x': 'launch', 'Bus Interface_x': 'bus',
                            'clock_x': 'clock', 'L2 Cache(MB)_x': 'L2 Cache(MB)', 'memory_clock_x': 'memory_clock',
                            'Memory Size (GB)_x': 'Memory Size (GB)', 'Memory Bus type_x': 'Memory Bus type',
                            'memory_size_x': 'memory_size', 'memory_bus_x': 'memory_bus', 'tdp_x': 'tdp',
                            'memory_bandwidth_x': 'memory_bandwidth', 'memory_bus_width_x': 'memory_bus_width',
                            'Transistors (billion)_x': 'Transistors (billion)',}
        self.fullDrop = ['TDP (Watts)_x', 'Core config1', 'Launch_y', 'codes2', 'Unnamed: 1_level_0 Launch',
                         'Unnamed: 1_level_0 Unnamed: 1_level_1', 'Features_y', 'hw_model_y', 'clock_y', 'Bus interface',
                         'Unnamed: 4_level_0 Unnamed: 4_level_1', 'Unnamed: 5_level_0 Unnamed: 5_level_1',
                         'Memory Size (GB)_y', 'TDP (Watts)_y', 'memory_bandwidth_y',  'memory_bus_width_y', 'memory_bus_y', 'memory_size_y',
                         'L2 Cache(MB)_y', 'Process_y', 'Notes, form factor Unnamed: 18_level_2', 'Features_x',
                         'MFLOPS FP32', 'MFLOPSFP32', 'Features', 'memory_clock_y', 'tdp_y',
                         'Latest API support Direct3D', 'Latest API support OpenGL', 'Transistors (billion)_y',
                         'Latest supported API version Direct3D', 'Latest supported API version OpenGL',
                         'Latest supported API version Other', 'Latest supported API version Vulkan',]
        self.tableType = 'other'
        self.df = self.process(pd.read_html(uri), folder)

    def process(self, dfRaw, folder, fileTemplate='nvidia'):
        """Perform the initial export and data normalization."""
        df = {'models': None, 'features': None, 'features1': None, 'architecture': None, 'technology': None, 'other': None}
        for num in range(len(dfRaw)):
            # print(f'ID: {num}')
            results = self.cleanup(copy(dfRaw[num]))
            if df[self.tableType] is None:
                df[self.tableType] = results
            else:
                if 'Company' in str(results.columns) or 'Key people' in str(results.iloc[0].values) or 'Company' in str(results.iloc[0].values):
                    print('Skip:', self.tableType, results)
                else:
                    df[self.tableType] = pd.concat([df[self.tableType], results], join='outer', ignore_index=True).fillna(0)
        # models = copy(df['models'])
        # for heading in list(models.columns):
        #     models[heading] = self.stripColumn(models[heading])
        # models['hw_model'] = self.stripColumn(models['hw_model'], key='(')
        # models['hw_model'] = self.stripColumn(models['hw_model'], key='*')
        # df['models'] = models
        df['models'] = self.cleanHeader(copy(df['models']), columnName='hw_model', keys=['(', '*'])
        # architecture = copy(df['architecture'])
        # for heading in list(architecture.columns):
        #     architecture[heading] = self.stripColumn(architecture[heading])
        # df['architecture'] = architecture
        df['architecture'] = self.cleanHeader(copy(df['architecture']))
        models = self.splitHw_model(copy(df['models']))
        models['Launch'] = [self.convertDate(models['Launch'].iloc[idx]) for idx in range(len(models))]
        models['Code name'] = [models['Code name'].iloc[idx].replace('2x', '').strip() for idx in range(len(models))]
        df['models'] = models
        self.df1 = df
        architecture = copy(df['architecture'])
        architecture['Launch'] = [self.convertDate(architecture['Launch'].iloc[idx]) for idx in range(len(architecture))]
        architecture = self.splitArch(architecture)
        architecture['codes'], architecture['codes2'] = self.mergeArchCode(df, 'architecture',
                                                                                       ['Code name(s)', # 'chips',
                                                                                        'Chips'])
        df['architecture'] = architecture
        df['full'] = df['models'].merge(df['features'], how='left', on='hw_model').fillna(0)
        df['full'] = df['full'].merge(df['architecture'], how='left', left_on='Code name', right_on='codes').fillna(0)
        df['full'] = df['full'].rename(columns=self.fullColumns)
        df['full'] = df['full'].drop(columns=self.fullDrop)

        for key in list(df.keys()):
            if df[key] is not None:
                df[key].to_csv(f'{folder}/{fileTemplate}_{key}.csv', index=False)
        return df

    def cleanHeader(self, df, columnName=None, keys=None):
        for heading in list(df.columns):
            df[heading] = self.stripColumn(df[heading])
        if columnName is not None:
            for key in keys:
                df['hw_model'] = self.stripColumn(df['hw_model'], key=key)
        return df

    def splitArch(self, df):
        model = []
        architecture = []
        for idx in range(len(df)):
            field = df.iloc[idx]['architecture'].split('(')
            if len(field) == 1:
                architecture.append(field[0])
                model.append(0)
            else:
                architecture.append(field[1].replace(')', ''))
                model.append(field[0])
        df['architecture'] = architecture
        df['model'] = model
        return df

    def mergeArchCode(self, df, table, keys):
        codes = []
        options = []
        for inx in range(len(df[table])):
            code = '0'
            option = []
            cellArr = []
            for row in keys:
                field = df[table][row].iloc[inx]
                charKey = ('(', 0) if '(' in field else (' ', 1)
                if len(df[table][row].iloc[inx]) > 0:
                    cellArr = df[table][row].iloc[inx].split(charKey[0])
                    if cellArr[0] != "0":
                        code = cellArr[charKey[1]].replace('2x', '').strip()
                        break
            for i in range(1, len(cellArr)):
                option.append(cellArr[i].replace(')', '').strip())
            codes.append(code)
            options.append(option)
        return [codes, options]

    def convertDate(self, strDate):
        strDate = str(strDate).split('/')[0].strip().replace('\xa0', ' ')
        if strDate in ['0', '0.0', '?', 'Unknown', 'Unlaunched']:
            return '1995-01-01'
        for key in ['(PCIe)', 'AGP', '(']:
            if key in strDate:
                strDate = strDate.split(key)[0].strip()
        arrDate = str(strDate).split(' ')
        arrDate[0] = arrDate[0].replace(',', '')
        match len(arrDate):
            case 1:
                return f'{strDate}-01-01'
            case 2:
                if len(arrDate[0]) == 3:
                    arrDate[0] = months[arrDate[0]]
                strDate = f'{arrDate[0]} 01, {arrDate[1]}'
            case 3:
                if len(arrDate[0]) == 3:
                    strDate = f'{months[arrDate[0]]} {arrDate[1].replace(",", "")}, {arrDate[2]}'
            case _:
                if len(arrDate) > 3:
                    strDate = f'{arrDate[0]} {arrDate[1].replace(",", "")}, {arrDate[2][:4]}'
        return datetime.strptime(strDate, '%B %d, %Y').strftime('%Y-%m-%d')

    def stripColumn(self,column, key='[', position=0):
        "Clear extra character substrings by returning the desired array position."
        return [str(cell).split(key)[position].strip() for cell in column]

    def splitHw_model(self, df, key='+'):
        for idx in range(len(df)):
            row = copy(df.iloc[idx])
            if key in row['hw_model']:
                models = row['hw_model'].split(key)
                df.iloc[idx]['hw_model'] = models[0].strip()
                pd.concat([df, row.to_frame().T], ignore_index=True)
        return df

    def normalizeHeader(self, columns):
        """Headers are 1-3 rows, detect and merge into one row."""
        headers = list(columns)
        return self.mergeRows(headers)

    def mergeRows(self, headers):
        """Called by normaizeHeader to identify and merge the header rows."""
        mergedRow = []
        for num in range(len(headers)):
            skipRow = False
            for cellNum in range(len(headers[0])):
                cell = headers[num][cellNum].split('[')[0].strip()
                if cell == 'Units' and cellNum == 0:
                    skipRow = True
                if cellNum == 0:
                    mergedRow.append(cell)
                elif cell not in mergedRow[num] and skipRow is False:
                    mergedRow[num] = f'{mergedRow[num]} {cell}'
        return mergedRow

    def findHeader(self, df):
        if len(df.columns) > 2:
            headers = []
            for num in range(len(df.columns)):
                headers.append([df.iloc[0][num].replace('Model',
                                                        'hw_model'), df.iloc[1][num].replace('Model', 'hw_model')])
            df.columns = self.mergeRows(headers)
            df = df.drop(index=[0, 1])
            self.tableType = 'models'
            return df
        elif 'Company' in str(df.columns):
            self.tableType = 'other'
            return df
        elif 'Key people' in df.iloc[0][0]:
            self.tableType = 'other'
            return df
        elif 'people' not in df.iloc[0][0]:
            df.columns = ['category', 'Features']
            self.tableType = 'technology'
            return df
        else:
            self.tableType = 'other'
            return df

    def cleanup(self, df):
        """Detect if a fotter exists and remove if True."""
        self.tableType = None
        if type(df.columns[0]) is not np.int64:
            if 'Features' in list(df.columns)[1]:  # or 'List of GPUs' in list(df.columns)[1]:
                self.tableType = 'features'
        elif type(df.columns[0]) is np.int64 and 'people' in str(df.iloc[0][0]):
            self.tableType = 'other'
        elif 'Key people' in str(df.iloc[0][0]) or 'Company' in str(df.iloc[0][0]):
            self.tableType = 'other'
        else:
            return self.findHeader(df)
        if 'MultiIndex' in str(type(df.columns)):
            df.columns = self.normalizeHeader(df.columns)
            if df.iloc[-1:, 0].values[0] == 'Model':
                df = df.head(-2)
        if type(list(df.columns)[0]) is not np.int64:
            df = df.rename(columns=self.cleanColumns)
        if 'Features' not in list(df.columns) and 'Code name' not in list(df.columns):
            df['Features'] = 'Yes'
        if self.tableType is None:
            self.getType(df.columns)
        return df.fillna(0)

    def getType(self, columns):
        """Determine the category for a table."""
        header = list(columns)
        # print(self.tableType)
        if 'architecture' in header:
            self.tableType = 'architecture'
        elif 'Archi-' in header:
            self.tableType = 'architecture'
        elif 'Micro-' in header:
            self.tableType = 'architecture'
        elif 'Model Units' in header:
            self.tableType = 'architecture'
        elif 'Company' in header:
            print('Found company')
            self.tableType = 'other'
        elif 'category' in header or 'other_products' in header or 'software_technologies' in header:
            self.tableType = 'technology'
        elif 'Features' in header and 'gpu' in header:
            # print('f1', header)
            if 'Launch' in header:
                self.tableType = 'models'
            elif 'software_technologies' or 'other_products' in header:
                self.tableType = 'other'
            else:
                self.tableType = 'features1'
        elif 'Code name' in header or 'Code name(s)' in header or 'clock' in header:
            self.tableType = 'models'
        else:
            self.tableType = 'other'


if __name__ == "__main__":
    folder = 'nvidia'
    if importNvidia is True:
        source = 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'
        nvidia = NvidiaImport(folder, source)
    else:
        full = nvidiaLoader(folder)
        header = nvidiaHeader(folder)
    # dfNvidia = process(pd.read_html(source), folder)
    # dfRaw = pd.read_html(source)
    # df = pd.read_csv(f'{folder}/nvidia_{num}.csv')
    # df = df.iloc[1:].set_axis(df.columns + ' ' + df.iloc[0], axis=1)
