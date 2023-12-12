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
from datetime import datetime

num = 0

# cleanColumns = {'Model Model': 'model', 'Launch Launch': 'launch', 'Code name Code name': 'code_name',
#                   'Fab (nm)[2] Fab (nm)[2]': 'fsb (nm)',
#                   'Transistors (million) Transistors (million)': 'Transistors(million)',
#                   'Die size (mm2) Die size (mm2)': 'die', 'Bus interface Bus interface': 'bus_interface',
#                   'Core clock (MHz) Core clock (MHz)': 'clock',
#                   'Memory clock (MHz) Memory clock (MHz)': 'Memory_clock(MHz)',
#                   'Core config[a] Core config[a]': 'core_config', 'Fillrate.1 MPixels/s': 'Fillrate MPixels/s',
#                   'Fillrate.2 MTexels/s': 'Fillrate MTexels/s', 'Fillrate.3 MVertices/s': 'Fillrate MVertices/s',
#                   'Memory Size (MB)': 'memory', 'Memory.1 Bandwidth (GB/s)': 'memory_bandwidth(GB/s)',
#                   'Memory.2 Bus type': 'memory_bus_type', 'Memory.3 Bus width (bit)': 'memory_bus_width(bit)',
#                   'MFLOPS FP32 MFLOPS FP32': 'mflops_fp32', 'Latest API support Direct3D': 'api_support_direct3d',
#                   'Latest API support.1 OpenGL': 'api_support_opengl'}

months = {'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April', 'May': 'May', 'Jun': 'June',
          'Jul': 'July', 'Aug': 'August', 'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'}

class NvidiaImport:
    """Export Nvidia data from wikipedia and import into a pandas dataframe."""

    def __init__(self, folder, uri):
        self.cleanColumns = {'Model': 'hw_model', 'Model': 'hw_model', 'Core clock (MHz)': 'clock', 'Model(Architecture)': 'architecture',
                             'Archi-tecture': 'architecture', 'Model Units': 'model',
                             'Micro-architecture Unnamed: 1_level_2': 'architecture',
                             'Launch Unnamed: 2_level_2': 'Launch', 'Chips Unnamed: 3_level_2': 'chips',
                             'Core clock(MHz) Unnamed: 4_level_2': 'clock',
                             'Shaders Cuda cores(total) Unnamed: 5_level_2': 'shaders_cuda_cores',
                             'Shaders Base clock (MHz)': 'shaders_clock',
                             'Shaders Max boostclock (MHz)': 'shaders_max_clock',
                             'Memory Bus type Unnamed: 8_level_2': 'memory_bus',
                             'Memory Bus width(bit) Unnamed: 9_level_2': 'memory_bus_width',
                             'Memory Size(GB) Unnamed: 10_level_2': 'memory_size',
                             'Memory Clock(MT/s) Unnamed: 11_level_2': 'memory_clock',
                             'Memory Bandwidth(GB/s) Unnamed: 12_level_2': 'memory_bandwidth',
                             'Processing power (GFLOPS) Half precisionTensor Core FP32 Accumulate Unnamed: 13_level_2': 'processing(GFLOPS)_Half_precision_tensor',
                             'Processing power (GFLOPS) Single precision(MAD or FMA) Unnamed: 14_level_2': 'processing(GFLOPS)_single_precision',
                             'Processing power (GFLOPS) Double precision(FMA) Unnamed: 15_level_2': 'processing(GFLOPS)_double_precision',
                             'CUDAcomputecapability Unnamed: 16_level_2': 'cuda_compute_capability',
                             'TDP(watts) W': 'tdp', 'form factor Unnamed: 18_level_2': 'form_factor',
                             'GeForce (List of GPUs)': 'Features', 'GeForce (List of GPUs).1': 'gpu',
                             'Other products': 'Features', 'Other products.1': 'other_products',
                             'Software and technologies': 'Features', 'Features nFiniteFX II Engine': 'Features',
                             'Software and technologies.1': 'software_technologies',
                             # 'vteNvidia': 'Features',
                             # 'vteNvidia.1': 'gpu',
                             'vteGraphics processing unit': 'category',
                             'vteGraphics processing unit.1': 'Features'}
        self.cleanDrop = ['Unnamed: 4_level_0 Unnamed: 4_level_1', 'Unnamed: 5_level_0 Unnamed: 5_level_1']
        self.tableType = 'other'
        self.df = self.process(pd.read_html(uri), folder)

    def process(self, dfRaw, folder, fileTemplate='nvidia'):
        """Perform the initial export and data normalization."""
        df = {'models': None, 'features': None, 'features1': None, 'architecture': None, 'technology': None, 'other': None}
        for num in range(len(dfRaw)):
            print(f'ID: {num}')
            results = self.cleanup(copy(dfRaw[num]))
            if df[self.tableType] is None:
                df[self.tableType] = results
            else:
                if 'Company' in str(results.columns) or 'Key people' in str(results.iloc[0].values) or 'Company' in str(results.iloc[0].values):
                    print('Skip:', self.tableType, results)
                else:
                    df[self.tableType] = pd.concat([df[self.tableType], results], join='outer', ignore_index=True).fillna(0)
        for heading in list(df['models'].columns):
            df['models'][heading] = self.stripColumn(df['models'][heading])
        df['models']['hw_model'] = self.stripColumn(df['models']['hw_model'], key='(')
        df['models']['hw_model'] = self.stripColumn(df['models']['hw_model'], key='*')
        for heading in list(df['architecture'].columns):
            df['architecture'][heading] = self.stripColumn(df['architecture'][heading])
        df['models'] = self.splitHw_model(copy(df['models']))
        df['models']['Launch'] = [self.convertDate(df['models']['Launch'].iloc[idx]) for idx in range(len(df['models']))]
        df['architecture']['Launch'] = [self.convertDate(df['architecture']['Launch'].iloc[idx]) for idx in range(len(df['architecture']))]
        df['architecture'] = self.splitArch(df['architecture'])
        df['architecture']['codes'], df['architecture']['codes2'] = self.mergeArchCode(df, 'architecture',
                                                                                       ['Code name(s)', 'chips',
                                                                                        'Chips'])

        df['full'] = df['models'].merge(df['features'], how='left', on='hw_model').fillna(0)
        for key in list(df.keys()):
            if df[key] is not None:
                df[key].to_csv(f'{folder}/{fileTemplate}_{key}.csv', index=False)
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
                # print(inx, row, field, charKey)
                if len(df[table][row].iloc[inx]) > 0:
                    cellArr = df[table][row].iloc[inx].split(charKey[0])
                    # print(cellArr)
                    if cellArr[0] != "0":
                        code = cellArr[charKey[1]]
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
        # print(strDate)
        match len(arrDate):
            case 1:
                return f'{strDate}-01-01'
            case 2:
                if len(arrDate[0]) == 3:
                    arrDate[0] = months[arrDate[0]]
                strDate = f'{arrDate[0]} 01, {arrDate[1]}'
            case 3:
                # arrDate[2] = arrDate[2][:4]
                if len(arrDate[0]) == 3:
                    strDate = f'{months[arrDate[0]]} {arrDate[1].replace(",", "")}, {arrDate[2]}'
            case _:
                if len(arrDate) > 3:
                    strDate = f'{arrDate[0]} {arrDate[1].replace(",", "")}, {arrDate[2][:4]}'
        # print(datetime.strptime(strDate, '%B %d, %Y').strftime('%Y-%m-%d'))
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
        # print('Merged:', mergedRow)
        return mergedRow

    def findHeader(self, df):
        # print('Missing Header:', len(df.columns))
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
            # print('f', list(df.columns), df.iloc[0])
            df.columns = ['category', 'Features']
            self.tableType = 'technology'
            return df
        else:
            self.tableType = 'other'
            return df

    def cleanup(self, df):
        """Detect if a fotter exists and remove if True."""
        self.tableType = None
        # print(list(df.columns))
        if type(df.columns[0]) is not np.int64:
            if 'Features' in list(df.columns)[1]: # or 'List of GPUs' in list(df.columns)[1]:
                self.tableType = 'features'
        elif type(df.columns[0]) is np.int64 and 'people' in str(df.iloc[0][0]):
            # print('People', df)
            self.tableType = 'other'
        elif 'Key people' in str(df.iloc[0][0]) or 'Company' in str(df.iloc[0][0]):
            # print('Key People', df)
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
        print(self.tableType)
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
            print('f1', header)
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
    source = 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'
    nvidia = NvidiaImport(folder, source)
    # dfNvidia = process(pd.read_html(source), folder)
    # dfRaw = pd.read_html(source)
    # df = pd.read_csv(f'{folder}/nvidia_{num}.csv')
    # df = df.iloc[1:].set_axis(df.columns + ' ' + df.iloc[0], axis=1)
