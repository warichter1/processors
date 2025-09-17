# CPU.py Documentation

## Overview

The `cpu.py` module provides a comprehensive framework for processing and analyzing CPU/processor data from multiple sources, primarily Intel and AMD. It handles data import, normalization, and conversion from various formats into a standardized database format suitable for analysis and comparison.

## Table of Contents

1. [Class Overview](#class-overview)
2. [Dependencies](#dependencies)
3. [Data Sources](#data-sources)
4. [Class Methods](#class-methods)
5. [Usage Examples](#usage-examples)
6. [Data Flow](#data-flow)
7. [Configuration](#configuration)

## Class Overview

### `Processors`

The main class that handles CPU data processing and management. It provides methods to:
- Import and process Intel processor data from ARK exports
- Import and process AMD processor data from CSV files
- Load and merge base processor databases
- Normalize processor names and specifications
- Convert date formats to EPOCH timestamps
- Map processor models to family IDs

## Dependencies

### Required Packages
- `pandas` - Data manipulation and analysis
- `copy` - Object copying utilities

### Local Modules
- `cpuConf` - Configuration module containing:
  - `amdRename`: Column mapping for AMD data
  - `amdDrop`: Columns to drop from AMD data
  - `intelRename`: Column mapping for Intel data
  - `intelDrop`: Columns to drop from Intel data

### Data Files Structure
```
cpudb/
├── processor_family.csv
├── microarchitecture.csv
├── manufacturer.csv
├── technology.csv
├── code_name.csv
├── gate_delay.csv
├── mips_est.csv
├── processor.csv
├── spec_int2006.csv
├── spec_int2000.csv
├── spec_int1995.csv
├── spec_int1992.csv
└── AMDcpu.csv

Intel/
├── [Intel ARK export files]
└── intel.txt (list of files to process)
```

## Data Sources

### Intel Data
- **Source**: Intel ARK export files (CSV format)
- **Format**: Records as columns, transposed to rows
- **ID Range**: 6000+
- **Processing**: Handles multiple files listed in `intel.txt`

### AMD Data
- **Source**: `AMDcpu.csv` file
- **Format**: Standard CSV with records as rows
- **ID Range**: 4000+
- **Processing**: Direct CSV import with column mapping

### Base Processor Database
- **Sources**: Multiple SPEC benchmark files (1992, 1995, 2000, 2006)
- **Merging**: Outer joins on processor_id
- **Date Conversion**: All date fields converted to EPOCH format

## Class Methods

### Constructor: `__init__(self)`

Initializes the Processors class with default configuration.

**Attributes:**
- `sourceDir`: Directory containing processor database files ('cpudb')
- `dateCol`: List of date columns for processing
- `filter`: Dictionary mapping date abbreviations to standardized formats
- `normalize`: List of strings to remove from processor names
- `lookup`: Dictionary containing loaded lookup tables

### Data Loading Methods

#### `loadtables(self)`
Loads all lookup tables from the cpudb directory.

**Tables Loaded:**
- processor_family
- microarchitecture
- manufacturer
- technology
- code_name
- gate_delay
- mips_est

#### `loadFile(self, filename)`
Utility method to load CSV files with pandas configuration.

**Parameters:**
- `filename` (str): Path to the CSV file

**Returns:**
- `pandas.DataFrame`: Loaded data

### Date Processing Methods

#### `datetime_to_epoch(self, ser)`
Converts datetime series to EPOCH timestamps, handling NaT values.

**Parameters:**
- `ser` (pandas.Series): Series containing datetime values

**Returns:**
- `pandas.Series`: EPOCH timestamps (seconds since 1970-01-01)

**Features:**
- Handles NaT (Not a Time) values without converting to large negative numbers
- Converts to Int64 to maintain null handling

#### `fixDate(self, cell)`
Normalizes date formats for consistent processing.

**Parameters:**
- `cell` (str/datetime): Date value to normalize

**Returns:**
- `str/datetime`: Normalized date value

**Normalization Rules:**
- Removes spaces and apostrophes
- Converts to lowercase
- Handles OEM date formats
- Maps quarter abbreviations (q1, q2, etc.) to actual dates
- Maps month names to dates

### Processor Identification Methods

#### `getProcessFamilyId(self, row, vendor, label, fields=['hw_model'])`
Identifies processor family ID by matching model names.

**Parameters:**
- `row` (pandas.Series): Processor record
- `vendor` (pandas.DataFrame): Vendor-specific processor family data
- `label` (str): Column name for hardware model
- `fields` (list): List of fields to search in

**Returns:**
- `int/None`: Processor family ID or None if not found

**Matching Logic:**
- Removes trademark symbols and normalizes text
- Special handling for AMD Ryzen and EPYC processors
- Special handling for AMD A-Series processors
- Fallback to hardware model matching

#### `getFamilyId(self, df, manufacturerId, family="./cpudb/processor_family.csv", label='hw_model.spec_int2k6')`
Adds processor family IDs to a dataframe.

**Parameters:**
- `df` (pandas.DataFrame): Processor data
- `manufacturerId` (int): Manufacturer ID (1=AMD, 9=Intel)
- `family` (str): Path to processor family CSV
- `label` (str): Column name for model matching

**Returns:**
- `pandas.DataFrame`: DataFrame with added processor_family_id column

### Import Methods

#### `importIntel(self, filename, folder='Intel', manufacturerId=9)`
Imports and processes Intel processor specifications.

**Parameters:**
- `filename` (str): File containing list of Intel files to process
- `folder` (str): Directory containing Intel files
- `manufacturerId` (int): Intel manufacturer ID (default: 9)

**Returns:**
- `pandas.DataFrame`: Processed Intel processor data

**Processing Steps:**
1. Reads file list from specified file
2. Processes each Intel ARK export file
3. Concatenates all data
4. Adds test_sponsor field
5. Maps to processor families
6. Normalizes dates

#### `importAmd(self, filename, manufacturerId=1, family="./cpudb/processor_family.csv")`
Processes AMD processor records.

**Parameters:**
- `filename` (str): AMD CSV filename
- `manufacturerId` (int): AMD manufacturer ID (default: 1)
- `family` (str): Path to processor family CSV

**Returns:**
- `pandas.DataFrame`: Processed AMD processor data

**Processing Steps:**
1. Loads AMD CSV file
2. Drops unnecessary columns
3. Renames columns using amdRename mapping
4. Assigns manufacturer ID and processor IDs (4000+)
5. Adds test_sponsor field
6. Maps to processor families
7. Normalizes dates

#### `processIntelFile(self, filename, indexStart=0)`
Processes individual Intel ARK export files.

**Parameters:**
- `filename` (str): Intel file to process
- `indexStart` (int): Starting index for processor IDs

**Returns:**
- `pandas.DataFrame`: Processed Intel data

**Processing Steps:**
1. Reads CSV skipping header rows
2. Transposes data (records as columns → records as rows)
3. Renames columns using intelRename mapping
4. Drops unnecessary columns
5. Assigns processor IDs (6000+)

### Specialized Parsing Methods

#### `getRyzen(self, cell)`
Parses AMD Ryzen processor model names.

**Parameters:**
- `cell` (str): Processor name string

**Returns:**
- `str`: Standardized Ryzen model name

**Parsing Logic:**
- Detects Radeon Graphics variants
- Identifies Threadripper models
- Handles PRO variants
- Extracts generation codes

#### `getASeries(self, cell)`
Parses AMD A-Series processor model names.

**Parameters:**
- `cell` (str): Processor name string

**Returns:**
- `str`: Standardized A-Series model name

**Parsing Logic:**
- Extracts series number
- Identifies APU variants
- Handles Radeon integration

### Database Loading Methods

#### `loadBaseProcessors(self)`
Loads and merges base processor database with SPEC benchmark data.

**Returns:**
- `pandas.DataFrame`: Merged processor database

**Merging Process:**
1. Loads base processor table
2. Loads SPEC benchmark tables (2006, 2000, 1995, 1992)
3. Performs outer joins on processor_id
4. Converts date columns to EPOCH format
5. Calculates performance metrics
6. Handles missing values

**Calculated Fields:**
- `max_clock`: Uses clock speed when max_clock is missing
- `perfnorm`: Performance per watt (basemean.spec_int2k6 / tdp)

## Usage Examples

### Basic Usage

```python
# Initialize the processor handler
processors = Processors()

# Import Intel data
intel_data = processors.importIntel('intel.txt')

# Import AMD data
amd_data = processors.importAmd("AMDcpu.csv")

# Load base processor database
base_data = processors.loadBaseProcessors()
```

### Custom Configuration

```python
# Initialize with custom source directory
processors = Processors()
processors.sourceDir = 'custom_cpudb'

# Load specific lookup table
family_data = processors.loadFile('./cpudb/processor_family.csv')

# Process dates manually
normalized_date = processors.fixDate("Q1 2023")  # Returns "1/1/2023"
```

### Data Analysis

```python
# Load all data
processors = Processors()
intel_data = processors.importIntel('intel.txt')
amd_data = processors.importAmd("AMDcpu.csv")
base_data = processors.loadBaseProcessors()

# Analyze performance metrics
high_performance = base_data[base_data['perfnorm'] > base_data['perfnorm'].mean()]

# Filter by manufacturer
intel_processors = base_data[base_data['manufacturer_id'] == 9]
amd_processors = base_data[base_data['manufacturer_id'] == 1]
```

## Data Flow

```
Intel ARK Files → processIntelFile() → importIntel() → Standardized DataFrame
AMD CSV File → importAmd() → Standardized DataFrame
Base Database → loadBaseProcessors() → Merged DataFrame with SPEC data

All DataFrames include:
- Normalized processor names
- Standardized column names
- EPOCH timestamps
- Processor family IDs
- Performance metrics
```

## Configuration

### Date Filters
The `filter` dictionary maps common date abbreviations:
- `'q1'`, `'q2'`, `'q3'`, `'q4'`: Quarter abbreviations
- `'september'`: Month names
- `'0416'`: Custom date codes

### Text Normalization
The `normalize` list removes common strings from processor names:
- Intel® branding
- Processor/Processors suffixes
- Series/Family designations
- Trademark symbols

### Manufacturer IDs
- AMD: 1
- Intel: 9

### Processor ID Ranges
- AMD: 4000+
- Intel: 6000+

## Error Handling

- **Missing Files**: Methods handle missing CSV files gracefully
- **Date Parsing**: Invalid dates are preserved as-is
- **Model Matching**: Unmatched processors print warning messages
- **Missing Values**: Filled with 0 or appropriate defaults

## Performance Considerations

- **Memory Usage**: Large datasets are processed in chunks where possible
- **File I/O**: CSV files are loaded with optimized pandas settings
- **String Operations**: Processor name matching uses efficient string methods
- **Merging**: Outer joins preserve all data while handling missing values

## Notes

- The module expects specific file structures in `cpudb/` and `Intel/` directories
- Column mappings in `cpuConf.py` must match the actual data file formats
- SPEC benchmark data is merged using outer joins to preserve all processor records
- Date conversion handles various input formats but may require manual cleanup for unusual formats