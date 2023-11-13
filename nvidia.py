#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:48:46 2023

@author: wrichter

Nvidia processor builder script
"""

import pandas as pd
import numpy as np

num = 0

cleanColumns = {'Model Model': 'model', 'Launch Launch': 'launch', 'Code name Code name': 'code_name',
                  'Fab (nm)[2] Fab (nm)[2]': 'fsb (nm)',
                  'Transistors (million) Transistors (million)': 'Transistors(million)',
                  'Die size (mm2) Die size (mm2)': 'die', 'Bus interface Bus interface': 'bus_interface',
                  'Core clock (MHz) Core clock (MHz)': 'clock',
                  'Memory clock (MHz) Memory clock (MHz)': 'Memory_clock(MHz)',
                  'Core config[a] Core config[a]': 'core_config', 'Fillrate.1 MPixels/s': 'Fillrate MPixels/s',
                  'Fillrate.2 MTexels/s': 'Fillrate MTexels/s', 'Fillrate.3 MVertices/s': 'Fillrate MVertices/s',
                  'Memory Size (MB)': 'memory', 'Memory.1 Bandwidth (GB/s)': 'memory_bandwidth(GB/s)',
                  'Memory.2 Bus type': 'memory_bus_type', 'Memory.3 Bus width (bit)': 'memory_bus_width(bit)',
                  'MFLOPS FP32 MFLOPS FP32': 'mflops_fp32', 'Latest API support Direct3D': 'api_support_direct3d',
                  'Latest API support.1 OpenGL': 'api_support_opengl'}

class NvidiaImport:
    def __init__(self, folder, uri):
        self.df = self.process(pd.read_html(uri), folder)

    def process(self, dfRaw, folder, fileTemplate='nvidia'):
        df = []
        for num in range(len(dfRaw)):
            df.append(self.cleanup(dfRaw[num]))
            df[num].to_csv(f'{folder}/{fileTemplate}_{num}.csv', index=False)
        return df

    def normalizeHeader(self, columns):
        headers = list(columns)
        mergedRow = []
        for num in range(len(headers)):
            for cellNum in range(len(headers[0])):
                cell = headers[num][cellNum].split('[')[0].strip()
                if cellNum == 0:
                    mergedRow.append(cell)
                elif cell not in mergedRow[num]:
                    mergedRow[num] = f'{mergedRow[num]} {cell}'
        return mergedRow

    def cleanup(self, df):
        if 'MultiIndex' in str(type(df.columns)):
            df.columns = self.normalizeHeader(df.columns)
            if df.iloc[-1:, 0].values[0] == 'Model':
                df = df.head(-2)
        return df



if __name__ == "__main__":
    folder = 'nvidia'
    source = 'https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units'
    nvidia = NvidiaImport(folder, source)
    # dfNvidia = process(pd.read_html(source), folder)
    # dfRaw = pd.read_html(source)
    # df = pd.read_csv(f'{folder}/nvidia_{num}.csv')
    # df = df.iloc[1:].set_axis(df.columns + ' ' + df.iloc[0], axis=1)
