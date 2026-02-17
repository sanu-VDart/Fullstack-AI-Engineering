"""
Feature configuration for C-MAPSS turbofan engine dataset.

The dataset contains 26 columns:
- Column 1: Unit number (engine ID)
- Column 2: Time in cycles
- Columns 3-5: Operational settings
- Columns 6-26: Sensor measurements
"""

# Column names for the C-MAPSS dataset
COLUMN_NAMES = [
    'unit_id',           # Engine unit number
    'cycle',             # Time in cycles
    'op_setting_1',      # Operational setting 1
    'op_setting_2',      # Operational setting 2
    'op_setting_3',      # Operational setting 3
    'sensor_1',          # Total temperature at fan inlet (T2)
    'sensor_2',          # Total temperature at LPC outlet (T24)
    'sensor_3',          # Total temperature at HPC outlet (T30)
    'sensor_4',          # Total temperature at LPT outlet (T50)
    'sensor_5',          # Pressure at fan inlet (P2)
    'sensor_6',          # Total pressure in bypass-duct (P15)
    'sensor_7',          # Total pressure at HPC outlet (P30)
    'sensor_8',          # Physical fan speed (Nf)
    'sensor_9',          # Physical core speed (Nc)
    'sensor_10',         # Engine pressure ratio (epr)
    'sensor_11',         # Static pressure at HPC outlet (Ps30)
    'sensor_12',         # Ratio of fuel flow to Ps30 (phi)
    'sensor_13',         # Corrected fan speed (NRf)
    'sensor_14',         # Corrected core speed (NRc)
    'sensor_15',         # Bypass Ratio (BPR)
    'sensor_16',         # Burner fuel-air ratio (farB)
    'sensor_17',         # Bleed Enthalpy (htBleed)
    'sensor_18',         # Demanded fan speed (Nf_dmd)
    'sensor_19',         # Demanded corrected fan speed (PCNfR_dmd)
    'sensor_20',         # HPT coolant bleed (W31)
    'sensor_21',         # LPT coolant bleed (W32)
]

# Sensors that show little to no variation (constant) - can be dropped
CONSTANT_SENSORS = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

# Sensors that are useful for RUL prediction
USEFUL_SENSORS = [s for s in COLUMN_NAMES if s.startswith('sensor_') and s not in CONSTANT_SENSORS]

# Operational settings
OPERATIONAL_SETTINGS = ['op_setting_1', 'op_setting_2', 'op_setting_3']

# All features to use for modeling
FEATURE_COLUMNS = OPERATIONAL_SETTINGS + USEFUL_SENSORS

# RUL capping value (improves model performance by focusing on degradation phase)
RUL_CAP = 125

# State classification thresholds
STATE_THRESHOLDS = {
    'healthy': 125,      # RUL > 125 cycles
    'degrading': 50,     # 50 < RUL <= 125 cycles
    'critical': 0        # RUL <= 50 cycles
}

# State labels
STATE_LABELS = ['healthy', 'degrading', 'critical']

# Rolling window sizes for feature engineering
ROLLING_WINDOWS = [5, 10, 20]

# Dataset information
DATASETS = {
    'FD001': {'train_trajectories': 100, 'test_trajectories': 100, 'conditions': 1, 'fault_modes': 1},
    'FD002': {'train_trajectories': 260, 'test_trajectories': 259, 'conditions': 6, 'fault_modes': 1},
    'FD003': {'train_trajectories': 100, 'test_trajectories': 100, 'conditions': 1, 'fault_modes': 2},
    'FD004': {'train_trajectories': 248, 'test_trajectories': 249, 'conditions': 6, 'fault_modes': 2},
}
