import pandas as pd
import numpy as np
import datetime
import time
import copy

sourceDir = 'cpudb'
dateCol = ['date', 'hw_avail.spec_int95']


def datetime_to_epoch(ser):
    """Don't convert NaT to large negative values."""
    if ser.hasnans:
        res = ser.dropna().astype('int64').astype('Int64').reindex(index=ser.index)
    else:
        res = ser.astype('int64')

    return res // 10**9


def fixDate(cell):
    """Normalize the date format to allow conversion to EPOCH format."""
    filter = {'q1': '1/1/', 'q2': '4/1/', '2q': '4/1/', 'q3': '9/1/', '3q': '9/1/', 'q4': '10/1/',
              'september': '9/1/', '0416': '4/1/16'}
    if pd.isnull(cell):
        return cell
    if type(cell) == str:
        cell = cell.lower().replace(' ', '').replace("'", '')
    else:
        return cell
    if 'oem:' in cell:
        return cell.split('oem:')[1]

    for key in filter:
        if key in cell:
            return cell.replace(key, filter[key])
    return cell


def getProcessFamilyId(row, vendor, label, fields=['hw_model']):
    """Process the rows in a processor file to convert to CPUDB format."""
    normalize = [' Intel®', 'Intel® ', ' Processors', ' Processor', ' (AF)', '®', ' Series', ' Scalable',
                 ' Family', ' Product', '™']
    # print(row)
    for field in fields:
        cell = row[field].replace('™', '')
        for key in normalize:
            cell = cell.replace(key, '')
        # print(cell)
        for modelId in vendor.index:
            model = vendor.loc[modelId]
            if type(row[label]) == str:
                hwModel = row[label].replace('™', '')
            else:
                hwModel = "NA"
            # cell = cell.strip()
            # print(model['name'], '||',  hwModel)
            if model['name'].lower() in cell.lower():
                return model['processor_family_id']
            if 'ryzen' in cell.lower() or 'epyc' in cell.lower():
                if model['name'] in getRyzen(cell):
                    return model['processor_family_id']
            if model['name'] in hwModel:
                return model['processor_family_id']
            if 'AMD' not in cell and model['name'] in getASeries(cell):
                return model['processor_family_id']
    print('Not found: ', row[field], '-', cell)
    return None


def getFamilyId(df, manufacturerId, family="./cpudb/processor_family.csv", label='hw_model.spec_int2k6'):
    """Add the proccessor family ID."""
    processorFamily = pd.read_csv(family)
    vendorFamily = copy.copy(processorFamily.loc[processorFamily['manufacturer_id'] == manufacturerId])
    df['processor_family_id'] = [getProcessFamilyId(df.loc[rowId], vendorFamily, label) for rowId in df.index]
    return df


def importIntel(filename, folder='Intel', manufacturerId=9):
    """Import and process Intel processor specs."""
    df = None
    with open(f"{folder}/{filename}", "r") as f:
        intelFiles = f.readlines()
    for file in intelFiles[0].replace('\n', '').split(','):
        print('Processing', file)
        if df is None:
            df = processIntelFile(file)
        else:
            df = pd.concat([df, processIntelFile(file, len(df))], join='outer', ignore_index=True)
    df = df.assign(test_sponsor='Intel')
    df.fillna(0, inplace=True)
    df = getFamilyId(df, manufacturerId, label='hw_model')
    df['date'] = [fixDate(cell) for cell in df['date']]
    return df


def importAmd(filename, manufacturerId=1, family="./cpudb/processor_family.csv"):
    """Process AMD records."""
    df = pd.read_csv(filename).drop(columns=amdDrop)
    df.rename(columns=amdRename, inplace=True)
    df.assign(manufacturer_id=manufacturerId, inplace=True)
    df['processor_id'] = [4000 + i for i in range(len(df))]
    df = df.assign(test_sponsor='AMD')
    df.fillna(0, inplace=True)
    df['date'] = [fixDate(cell) for cell in df['date']]
    return getFamilyId(df, manufacturerId)


def getRyzen(cell):
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


def getASeries(cell):
    """Parse an AMD A Series Processor record."""
    rowArr = cell.split(' ')
    name = f"{rowArr[0].split('-')[0]}-Series"
    if 'a' == cell[0].lower() and 'radeon' in cell.lower():
        return f'{name} APU'
    else:
        return name


def processIntelFile(filename, indexStart=0):
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


if __name__ == "__main__":
    amdRename = {'Model': 'hw_model', 'Unnamed: 0': 'processor_id',
                 'Line': 'hw_model.spec_int2k6', 'L1 Cache': 'l1_cache',
                 'L2 Cache': 'l2_cache', 'L3 Cache': 'l3_cache', 'Launch Date': 'date',
                 'Family': 'test_sponsor.spec_int2k6', 'Total Cores': 'hw_ncores', 'Total Threads': 'hw_threads',
                 '# of CPU Cores': 'hw_ncores', 'Memory Channels': 'memory channels',
                 '# of Threads': 'hw_nthreadspercore', 'Graphics Core Count': 'hw_gcores',
                 'Base Clock': 'clock', 'Max. Boost Clock ¹ ²': 'max_clock', 'System Memory Type': 'memory types',
                 'Default TDP': 'tdp', 'AMD Configurable TDP (cTDP)': 'ctdp', '*OS Support': 'os Support'}
    amdDrop = ['Thermal Solution PIB', 'Recommended Cooler', 'Thermal Solution MPK',
               'PCI Express® Version', 'CPU Socket', 'Socket Count', '1kU Pricing',
               'Unlocked for Overclocking', 'All Core Boost Speed', 'Unnamed: 1',
               'Product ID Tray', 'Product ID Boxed', 'Product ID MPK']

    intelRename = {'Product Collection': 'hw_model', 'Processor Number': 'test_sponsor.spec_int2k6',
                   'Processor Graphics ‡': 'graphics_model', 'System Memory Type': 'System Memory Specification',
                   'TDP': 'tdp', 'Configurable TDP-down': 'ctdp', 'Launch Date': 'date', 'Cache': 'l2_cache',
                   'Memory Types': 'memory types', 'Max # of Memory Channels': 'memory_channels',
                   'Vertical Segment': 'Platform', 'Processor Base Frequency': 'clock',
                   'Intel® Turbo Boost Technology 2.0 Frequency‡': 'max_clock', 'Graphics Name‡': 'graphics_mode',
                   'Total Cores': 'hw_ncores', 'Total Threads': 'total_threads',
                   '# of Performance-cores': 'performance_cores', '# of Efficiency-cores': 'Efficiency-cores',
                   'Platform': 'platform', '# of Displays Supported‡': 'max_displays',
                   'Intel® Turbo Boost Max Technology 3.0 Frequency ‡': 'turbo_boost_clock',
                   'Graphics Base Frequency': 'graphics_base', 'Graphics Max Dynamic Frequency': 'graphics_max',
                   'Graphics Output': 'graphics_out'}
    intelDrop = ['Recommended Customer Price', 'Essentials', 'nan', 'Use Conditions', 'Marketing Status',
                 'Servicing Status', 'End of Servicing Updates Date', 'Embedded Options Available',
                 '4K Support', 'Max Resolution (HDMI)‡', '# of QPI Links', 'Graphics Base Clock'
                 'Max Resolution (DP)‡', 'Max Resolution (eDP - Integrated Flat Panel)‡', 'Baseband Functions',
                 'RF Transceiver', 'RF Transceiver Functions', 'Protocol Stack', 'Intel® Smart Idle Technology',
                 'DirectX* Support', 'OpenGL* Support', 'Intel® Quick Sync Video', 'Max Refresh Rate',
                 'Intel® InTru™ 3D Technology', 'Intel® Clear Video HD Technology',
                 'Intel® Clear Video Technology', '# of Displays Supported ‡', 'Thermal Solution Specification',
                 'Package Size', 'TJUNCTION', 'Intel® Turbo Boost Technology ‡', 'Intel® Hyper-Threading Technology ‡',
                 'Intel® Transactional Synchronization Extensions', 'Intel® 64 ‡', 'Instruction Set',
                 'Instruction Set Extensions', 'Idle States', 'Enhanced Intel SpeedStep® Technology',
                 'Thermal Monitoring Technologies', 'Intel® Identity Protection Technology ‡',
                 'Intel® My WiFi Technology', 'Intel® Flex Memory Access', 'Intel® Smart Response Technology',
                 'Intel® Speed Shift Technology', 'Intel vPro® Eligibility ‡', 'Intel® AES New Instructions',
                 'Secure Key', 'Intel® Software Guard Extensions (Intel® SGX)', 'Graphics Base Clock', 'PCI Support',
                 'Intel® Memory Protection Extensions (Intel® MPX)', 'Intel® OS Guard', 'Supported FSBs',
                 'Performance-core Max Turbo Frequency', 'Efficient-core Max Turbo Frequency', '# of Efficient-cores',
                 'Intel® Trusted Execution Technology ‡', 'Execute Disable Bit ‡', 'Macrovision* License Required',
                 'Interfaces Supported', 'End of Interactive Support Date', 'Additional Information',
                 'Maximum High Bandwidth Memory (HBM)†', 'Integrated Intel® Omni-Path Architecture (Intel® OPA)',
                 'Max Resolution (VGA)‡', 'Integrated Wireless‡', '4G WiMAX Wireless Technology',
                 'Intel® Boot Guard', 'Intel® Stable IT Platform Program (SIPP)', 'Intel® Quick Resume Technology',
                 'Intel® Quiet System Technology', 'Intel® AC97 Technology', 'Intel® I/O Acceleration Technology',
                 'ECC Memory Supported ‡', 'FSB Parity', 'Expected Discontinuance', 'Processing Die Size',
                 'Intel® Virtualization Technology (VT-x) ‡', 'ECC Memory Supported   ‡',
                 'Intel® Virtualization Technology for Directed I/O (VT-d) ‡', 'OpenCL* Support',
                 'Multi-Format Codec Engines', 'Intel® VT-x with Extended Page Tables (EPT) ‡', 'Scalability',
                 'PCI Express Revision', 'PCI Express Configurations ‡', 'Intel® Turbo Boost Max Technology 3.0 ‡',
                 'Max # of PCI Express Lanes', 'Configurable TDP-up Frequency', 'Configurable TDP-up',
                 'Configurable TDP-down Frequency',      'Intel® Stable Image Platform Program (SIPP)',
                 'Intel® Threat Detection Technology (TDT)', 'Intel® Active Management Technology (AMT) ‡',
                 'Intel® Standard Manageability (ISM) ‡', 'Intel® One-Click Recovery ‡',
                 'Intel® Hardware Shield Eligibility ‡', 'Intel® Control-Flow Enforcement Technology',
                 'Intel® Total Memory Encryption - Multi Key', 'Intel® Total Memory Encryption',
                 'Mode-based Execute Control (MBE)', 'Intel® QuickAssist Software Acceleration',
                 'Intel® Virtualization Technology with Redirect Protection (VT-rp) ‡',
                 'Intel® On Demand Activation Model', 'Intel® Data Streaming Accelerator (DSA)',
                 'Intel® Advanced Matrix Extensions (AMX)', 'Intel® QuickAssist Technology (QAT)',
                 'Intel® Dynamic Load Balancer (DLB)', 'Graphics Burst Frequency', 'Warranty Period',
                 'Intel® Thermal Velocity Boost Temperature', 'Intel® Adaptive Boost Technology', 'Burst Frequency',
                 'Intel® Optane™ DC Persistent Memory Supported', 'Package Carrier', 'DTS Max',
                 'Intel® In-memory Analytics Accelerator (IAA)', 'Intel® vRAN Boost', 'Maximum Turbo Power',
                 'Default Maximum Enclave Page Cache (EPC) Size for Intel® SGX', 'Intel® Crypto Acceleration',
                 'Intel® Platform Firmware Resilience Support', 'Maximum Enclave Size Support for Intel® SGX',
                 'Intel® Speed Select Technology', 'Additional Information URL', 'Intel® UPI Speed',
                 'Intel® Instruction Replay Technology', 'VID Voltage Range', 'Intel SpeedStep® Max Frequency',
                 'Intel® Remote Platform Erase (RPE) ‡', 'Vulkan*  Support', 'H.264 Hardware Encode/Decode',
                 'H.265 (HEVC) Hardware Encode/Decode', 'MIPI SoundWire*', 'Intel® Adaptix™ Technology',
                 'Intel® Optane™ Memory Supported ‡', 'Intel® TSX-NI', '# of AVX-512 FMA Units',
                 'Intel® Volume Management Device (VMD)', 'Intel® Time Coordinated Computing (Intel® TCC)‡',
                 'Intel® Gaussian & Neural Accelerator', 'Intel® Thread Director', 'Maximum Assured Frequency',
                 'Intel® Deep Learning Boost (Intel® DL Boost)', 'Intel® Image Processing Unit',
                 'Intel® Smart Sound Technology', 'Intel® Wake on Voice', 'Intel® High Definition Audio',
                 'Intel® Thermal Velocity Boost', 'Secure Boot', 'Operating Temperature (Maximum)',
                 'Operating Temperature (Minimum)', 'TCASE', 'Direct Media Interface (DMI) Revision',
                 'Intel® Thunderbolt™ 4', 'Microprocessor PCIe Revision', 'Max # of DMI Lanes',
                 'Chipset / PCH PCIe Revision', 'Included Items', 'Product Brief', 'Description',
                 'Functional Safety (FuSa) Documentation Available', 'Minimum Assured Power',
                 'Intel® Thermal Velocity Boost Frequency', 'Minimum Assured Frequency', 'Maximum Assured Power', ]

    processor = pd.read_csv("./cpudb/processor.csv")
    specIntel2k23 = importIntel('intel.txt')
    specAmd2k23 = importAmd("./cpudb/AMDcpu.csv")
    specint2k6 = pd.read_csv("./cpudb/spec_int2006.csv")
    specint2k0 = pd.read_csv("./cpudb/spec_int2000.csv")
    specint95 = pd.read_csv("./cpudb/spec_int1995.csv")
    specint92 = pd.read_csv("./cpudb/spec_int1992.csv")

    df = processor.merge(specint2k6, on="processor_id", suffixes=(".proc", ".spec_int2k6"), how='outer')
    df = df.merge(specint2k0, on="processor_id", how='outer',
                  suffixes=(".spec_int2k6", ".spec_int2k0"))
    df = df.merge(specint95, on="processor_id", how='outer',
                  suffixes=(".spec_int2k0", ".spec_int95"))
    df = df.merge(specint92, on="processor_id", how='outer',
                  suffixes=(".spec_int95", ".spec_int92"))
    for field in dateCol:
        df[field] = datetime_to_epoch(pd.to_datetime(pd.Series(df[field])))
    df["max_clock"].fillna("clock", inplace=True)
    df.max_clock = df.clock.where(df.max_clock == 'clock', df.max_clock)
    df["perfnorm"] = df["basemean.spec_int2k6"]/df["tdp"]
    # df["perfnorm"] = df["perfnorm"].fillna(0)
    cpuStats = x = df.describe()
    # performance
    # df["perfnorm"] = df["basemean.spec_int2k6"]/df["tdp"]
    df["perfnorm"] = df["perfnorm"].fillna(0)
    spec95to2k0 = df[['basemean.spec_int2k0', 'basemean.spec_int95']].mean(axis=1)
    spec2k0to2k6 = df[['basemean.spec_int2k6', 'basemean.spec_int2k0']].mean(axis=1)
    no95 = df["basemean.spec_int95"].isna()
    no2k0 = df["basemean.spec_int2k0"].isna()
    no2k6 = df["basemean.spec_int2k6"].isna()
    scaleclk = min(df["max_clock"].values)
    # df["transistors"].fillna("na", inplace=True)
    # scaletrans = min(df["transistors"].value)
    scaletrans = cpuStats["transistors"]['min']
    scaletdp = cpuStats["tdp"]['min']
    scaleperf = cpuStats["basemean.spec_int2k6"]['min']
    scaleperfnorm = cpuStats["perfnorm"]['min']
    # scaletdp   <- min(all[["tdp"]], na.rm=TRUE)
    # scaleperf  <- min(all[["basemean.spec_int2k6"]], na.rm=TRUE)
    # scaleperfnorm  <- min(all[["perfnorm"]], na.rm=TRUE)
    df[no95, "basemean.spec_int95"]
