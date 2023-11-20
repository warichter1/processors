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


class NvidiaImport:
    """Export Nvidia data from wikipedia and import into a pandas dataframe."""

    def __init__(self, folder, uri):
        self.cleanColumns = {'Model': 'hw_model', 'Model': 'hw_model', 'Core clock (MHz)': 'clock', 'Model(Architecture)': 'architecture',
                             'Archi-tecture': 'architecture', 'Model Units': 'model',
                             'Micro-architecture Unnamed: 1_level_2': 'architecture',
                             'Launch Unnamed: 2_level_2': 'launch', 'Chips Unnamed: 3_level_2': 'chips',
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
                             # 'Features Video Processing Engine (VPE)': 'Video Processing Engine (VPE)',
                             # 'Features OpenEXR HDR': 'Features', 'Features TurboCache': 'TurboCache',
                             # 'Features PureVideo WMV9 Decoding': 'PureVideo WMV9 Decoding',
                             # 'Features Scalable Link Interface (SLI)': 'Scalable Link Interface (SLI)',
                             # 'Features Gamma-correct antialiasing': 'Features',
                             # 'Features 64-bit OpenEXR HDR': '64-bit OpenEXR HDR',
                             # 'Features Dual Link DVI': 'Dual Link DVI',
                             # 'Features ScalableLinkInterface(SLI)': 'Features', 'Features 3-WaySLI': '3-WaySLI',
                             # 'Features PureVideo HDwith VP1': 'PureVideo HDwith VP1',
                             # 'Features PureVideo 2 with VP2,BSP Engine, and AES128 Engine': 'PureVideo 2 with VP2,BSP Engine, and AES128 Engine',
                             # 'Features PureVideo 2 with VP2Engine: (BSP and 240 AES)': 'Features',
                             # 'Features PureVideo 3 with VP3,BSP Engine, and AES128 Engine': 'PureVideo 3 with VP3,BSP Engine, and AES128 Engine',
                             # 'Features PureVideo 4 with VP4': 'PureVideo 4 with VP4',
                             # 'Features Computeability': 'Computeability',
                             'Software and technologies.1': 'software_technologies', 'vteNvidia': 'Features',
                             'vteNvidia.1': 'gpu', 'vteGraphics processing unit': 'category',
                             'vteGraphics processing unit.1': 'Features'}
        self.cleanDrop = ['Unnamed: 4_level_0 Unnamed: 4_level_1', 'Unnamed: 5_level_0 Unnamed: 5_level_1']
        self.tableType = 'other'
        self.df = self.process(pd.read_html(uri), folder)

    def process(self, dfRaw, folder, fileTemplate='nvidia'):
        """Perform the initial export and data normaization."""
        df = {'models': [], 'features': [], 'architecture': [], 'other': []}
        # df = []
        for num in range(len(dfRaw)):
            print(f'ID: {num}')
            buffer = self.cleanup(copy(dfRaw[num]))
            df[self.tableType].append(buffer)
            buffer.to_csv(f'{folder}/{fileTemplate}_{num}.csv', index=False)
        return df

    def normalizeHeader(self, columns):
        """Headers are 1-3 rows, detect and merge into one row."""
        headers = list(columns)
        return self.mergeRows(headers)

    def mergeRows(self, headers):
        print('Premerge:', headers)
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
        print('Merged:', mergedRow)
        return mergedRow

    def findHeader(self, df):
        print('Missing Header:', len(df.columns))
        if len(df.columns) > 2:
            headers = []
            for num in range(len(df.columns)):
                headers.append([df.iloc[0][num].replace('Model',
                                                        'hw_model'), df.iloc[1][num].replace('Model', 'hw_model')])
            df.columns = self.mergeRows(headers)
            df = df.drop(index=[0, 1])
            self.tableType = 'models'
            return df
        else:
            self.tableType = 'other'
            return df

    def cleanup(self, df):
        """Detect if a fotter exists and remove if True."""
        self.tableType = None
        print(list(df.columns))
        if type(list(df.columns)[0]) is not int:
            if 'Features' in list(df.columns)[1] or 'List of GPUs' in list(df.columns)[1]:
                # if 'Features' not in list(df.columns):
                # df['Features'] = 'Yes'
                self.tableType = 'features'
        else:
            return self.findHeader(df)
        if 'MultiIndex' in str(type(df.columns)):
            df.columns = self.normalizeHeader(df.columns)
            if df.iloc[-1:, 0].values[0] == 'Model':
                df = df.head(-2)
        if type(list(df.columns)[0]) is not int:
            df = df.rename(columns=self.cleanColumns)
        if 'Features' not in list(df.columns) and 'Code name' not in list(df.columns):
            df['Features'] = 'Yes'
            # self.tableType = 'features
        if self.tableType is None:
            self.getType(df.columns)
        return df.fillna(0)

    def getType(self, columns):
        """Determine the category for a table."""
        header = list(columns)
        # print(header)
        if 'architecture' in header:
            self.tableType = 'architecture'
        elif 'Archi-' in header:
            self.tableType = 'architecture'
        elif 'Micro-' in header:
            self.tableType = 'architecture'
        elif 'Model Units' in header:
            self.tableType = 'architecture'
        elif 'Features' in header and 'Launch' not in header:
            self.tableType = 'features'
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
