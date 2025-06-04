#!/usr/bin/env python
"""
Penn State University - Amirmoez Jamaat
Purpose: Data assimilation with three modes: state+precip, state-only, or precip-only
"""

# =============================================================================
# IMPORTS
# =============================================================================
import sys
sys.path.append('../../')

import os
import json
import random
import datetime as dt
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import hydroDL
from hydroDL import utils
from hydroDL.data import camels
from hydroDL.post import plot, stat
from hydroDL.model import rnn, cnn, crit

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# GPU Configuration
TRAIN_GPU_ID = 0
TEST_GPU_ID = 1
torch.cuda.set_device(TRAIN_GPU_ID)

# Random Seed Configuration
RANDOM_SEED = 111111
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# ASSIMILATION MODE CONFIGURATION
# =============================================================================
# Choose assimilation mode:
# 'state_precip' - Assimilate both state variables and precipitation
# 'state_only' - Assimilate only state variables
# 'precip_only' - Assimilate only precipitation
ASSIMILATION_MODE = 'state_precip'  # Change to 'state_only' or 'precip_only'

# Window adjustment position (only used in 'state_precip' and 'precip_only' modes)

# For example: a=2 ::> adjsuter time
WINDOW_ADJUST_POSITION = 2

# Output verbosity

# 'none' - No output during processing (fastest)
OUTPUT_VERBOSITY = 'minimal'  # Change to 'none' for silent mode

# =============================================================================
# TIME CONFIGURATION
# =============================================================================
# Choose how to specify time period:


# 'calendar' - Use calendar dates (START_DATE, END_DATE)
TIME_MODE = 'calendar'

# Reference date for the dataset (beginning of the time series)
# This should match your data's starting date
REFERENCE_DATE = datetime(1989, 10, 1)  # October 1, 1989

# Time window for assimilation (total days in the dataset)
assimilation_time = 3651  # Approximately 10 years from 1989/10/01 to 1999/09/30

# Option 1: Specify using day indices (0-based)


# Option 2: Specify using calendar dates
START_DATE = datetime(1991, 9, 1)   # Example: November 5, 1991
END_DATE = datetime(1991, 12, 10)     # Example: February 13, 1992

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Batch Configuration
BATCH_SIZE = 531
START_BATCH = 0

# Model Hyperparameters
EPOCH = 100
HIDDEN_SIZE = 256
RHO = 365
SAVE_EPOCH = 5
ALPHA = 0.25
BUFFER_TIME = 365
N_MUL = 16
N_FEA = 14

# Training Configuration
WINDOW_SIZE = 5
WINDOWS_UPPER = 15 - WINDOW_SIZE
NUM_EPOCHS = 5
EVAL_WINDOW = 1

# Dynamic Parameters Configuration
TD_REP = [1, 3, 13]  # Index of dynamic parameters
TD_REP_S = [str(ix) for ix in TD_REP]
DY_DROP = 0.0  # Possibility to make dynamic become static
STA_IND = -1

# Model Options
PU_OPT = 0  # 0 for All; 1 for PUB; 2 for PUR
BUFF_OPT_ORI = 0
BUFF_OPT = 0

ROUTING = True
COMP_ROUT = False
COMP_WTS = False
P_CORR = None
ET_MOD = True

# Time Configuration
TEST_SEED = 111111

# Learning Rate Configuration
LR1 = 25.9

EPSILON = 1e-6

# =============================================================================
# DATE UTILITY FUNCTIONS
# =============================================================================

def date_to_index(date, reference_date=REFERENCE_DATE):
    """Convert a datetime object to day index from reference date."""
    delta = date - reference_date
    return delta.days

def index_to_date(index, reference_date=REFERENCE_DATE):
    """Convert a day index to datetime object."""
    return reference_date + timedelta(days=index)

def parse_date_string(date_string):
    """Parse date string in various formats to datetime object."""
    formats = [
        '%Y/%m/%d',
        '%Y-%m-%d',
        '%Y%m%d',
        '%m/%d/%Y',
        '%m-%d-%Y',
        '%d/%m/%Y',
        '%d-%m-%Y'
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse date string: {date_string}")

def get_assimilation_period():
    """Get start and end day indices based on TIME_MODE configuration."""

    start_day = date_to_index(START_DATE)
    end_day = date_to_index(END_DATE)

        # Validate dates
    if start_day < 0:
        raise ValueError(f"START_DATE {START_DATE} is before REFERENCE_DATE {REFERENCE_DATE}")
    if end_day > assimilation_time:
        raise ValueError(f"END_DATE {END_DATE} is beyond the data range (max {assimilation_time} days)")
    if start_day >= end_day:
        raise ValueError(f"START_DATE must be before END_DATE")

    print(f"Using calendar dates:")
    print(f"  Start: {START_DATE.strftime('%Y/%m/%d')} (day {start_day})")
    print(f"  End: {END_DATE.strftime('%Y/%m/%d')} (day {end_day})")



    return start_day, end_day

# =============================================================================
# FILE PATHS
# =============================================================================

# Input Data Paths
PARA_HBV_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/Daymet_3dyn_new_interface_HBVPara_531.npy"
PARA_ROUT_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/Daymet_3dyn_new_interface__RoutPara_531.npy"
HBV_FORCING_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/forcing_new_interface_daymet.npy"
RESULT_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/3dyn_prediction.npy"
STREAMFLOW_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/3dyn_obs.npy"

# State Variable Paths
SNOWPACK_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/Daymet_3dyn_SNOWMPACK_matrix_531.npy"
SM_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/Daymet_3dyn_SM_matrix_531.npy"
MELTWATER_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/Daymet_3dyn_MELTWATER_matrix_531.npy"
SUZ_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/Daymet_3dyn_SUZ_matrix_531.npy"
SLZ_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/Daymet_3dyn_SLZ_matrix_531.npy"
SWE_SCALER_FILE = "/mnt/sdb/fzr5082/Inputs_VD/DA_Data/SWE_scaler_531.npy"

# Output Paths
OUTPUT_DIR = '/mnt/sdb/fzr5082/Inputs_VD/DA_Data'
SAVE_DIR = "/mnt/sdb/fzr5082/output/"

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load all necessary data files."""
    # Load parameters
    all_basin_para_hbv1 = np.load(PARA_HBV_FILE)
    all_basin_para_rout1 = np.load(PARA_ROUT_FILE)

    # Slice parameters based on batch configuration
    all_basin_para_hbv = all_basin_para_hbv1[:, START_BATCH:BATCH_SIZE, :, :]
    all_basin_para_rout = all_basin_para_rout1[START_BATCH:BATCH_SIZE, :]

    # Load forcing data
    hbv_forcing = np.load(HBV_FORCING_FILE)
    hbv_forcing = hbv_forcing[START_BATCH:BATCH_SIZE, :, :]

    # Load streamflow data
    streamflow_prediction = np.load(RESULT_FILE)
    streamflow = np.load(STREAMFLOW_FILE)
    streamflow = streamflow[START_BATCH:BATCH_SIZE, :, :]
    streamflow_prediction = streamflow_prediction[START_BATCH:BATCH_SIZE, :, :]

    # Slice time dimension
    streamflow = streamflow[:, -assimilation_time:, :]
    streamflow_prediction = streamflow_prediction[:, -assimilation_time:, :]

    return (all_basin_para_hbv, all_basin_para_rout, hbv_forcing,
            streamflow, streamflow_prediction, all_basin_para_hbv1)

def load_state_variables():
    """Load state variables from files."""
    snowpack = np.load(SNOWPACK_FILE)
    sm = np.load(SM_FILE)
    meltwater = np.load(MELTWATER_FILE)
    suz = np.load(SUZ_FILE)
    slz = np.load(SLZ_FILE)


    return snowpack, sm, meltwater, suz, slz

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def save_states(states, filename="state_variables.pt"):
    """Save state variables to file."""
    torch.save(states, filename)

def prepare_state_tensors(snowpack, sm, meltwater, suz, slz):
    """Convert state variables to CUDA tensors and slice appropriately."""
    # Convert to tensors
    sm = torch.from_numpy(sm).cuda()
    snowpack = torch.from_numpy(snowpack).cuda()
    suz = torch.from_numpy(suz).cuda()
    meltwater = torch.from_numpy(meltwater).cuda()
    slz = torch.from_numpy(slz).cuda()

    # Slice based on batch and time
    sm_all = sm[-assimilation_time:, START_BATCH:BATCH_SIZE, :]
    snowpack_all = snowpack[-assimilation_time:, START_BATCH:BATCH_SIZE, :]
    suz_all = suz[-assimilation_time:, START_BATCH:BATCH_SIZE, :]
    meltwater_all = meltwater[-assimilation_time:, START_BATCH:BATCH_SIZE, :]
    slz_all = slz[-assimilation_time:, START_BATCH:BATCH_SIZE, :]

    return snowpack_all, sm_all, meltwater_all, suz_all, slz_all

def compute_evaluation_metrics(predictions, observations):
    """Compute evaluation metrics."""
    key_list = ['NSE', 'KGE', 'lowRMSE', 'highRMSE']

    # Convert to numpy if needed
    pred_array = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    obs_array = observations.cpu().numpy() if torch.is_tensor(observations) else observations

    # Squeeze dimensions
    pred_2d = np.squeeze(pred_array, axis=2)
    obs_2d = np.squeeze(obs_array, axis=2)

    # Compute statistics
    eva_dict = [stat.statError(np.swapaxes(pred_2d, 1, 0), np.swapaxes(obs_2d, 1, 0))]

    # Collect results
    data_box = []
    for i_s in range(len(key_list)):
        stat_str = key_list[i_s]
        temp = []
        for k in range(len(eva_dict)):
            data = eva_dict[k][stat_str]
            data = data[~np.isnan(data)]
            temp.append(data)
        data_box.append(temp)

    # Print results
    print(f"NSE: {np.nanmedian(data_box[0][0]):.4f}")
    print(f"KGE: {np.nanmedian(data_box[1][0]):.4f}")
    print(f"lowRMSE: {np.nanmedian(data_box[2][0]):.4f}")
    print(f"highRMSE: {np.nanmedian(data_box[3][0]):.4f}")

    return eva_dict, data_box

# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """Main data assimilation process."""

    # Get assimilation period
    print("\n" + "="*60)
    print("TIME CONFIGURATION")
    print("="*60)
    print(f"Reference date: {REFERENCE_DATE.strftime('%Y/%m/%d')}")
    print(f"Total data period: {assimilation_time} days")

    START_DAY, END_DAY = get_assimilation_period()
    NUM_DAYS = END_DAY - START_DAY

    print(f"Data assimilation days: {NUM_DAYS}")
    print(f"Output verbosity: {OUTPUT_VERBOSITY}")
    print("="*60 + "\n")

    # Load data
    print("Loading data...")
    (all_basin_para_hbv, all_basin_para_rout, hbv_forcing,
     streamflow, streamflow_prediction, all_basin_para_hbv1) = load_data()

    # Load state variables
    print("Loading state variables...")
    snowpack, sm, meltwater, suz, slz = load_state_variables()




    # Prepare state tensors
    snowpack_all, sm_all, meltwater_all, suz_all, slz_all = prepare_state_tensors(
        snowpack, sm, meltwater, suz, slz
    )

    # Prepare data for assimilation
    x_da = hbv_forcing[:, -assimilation_time:, :, ]
    x_da = np.swapaxes(x_da, 1, 0)
    x_da_adjusted = x_da.copy()

    y_da = streamflow[:, :, 0:1]
    y_da = np.swapaxes(y_da, 1, 0)

    yp_da = streamflow_prediction[:, :, 0:1]
    yp_da = np.swapaxes(yp_da, 1, 0)

    para_hbv = all_basin_para_hbv1[-assimilation_time:, :, :, :]

    # Prepare parameters


    # Convert routing weights to tensor
    rtwts = all_basin_para_rout[:, :]
    rtwts = torch.from_numpy(rtwts).float().cuda()

    # Initialize output tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_eval_values = torch.empty((NUM_DAYS), BATCH_SIZE - START_BATCH, 1).to(device)
    yp_w_da = torch.empty((NUM_DAYS), BATCH_SIZE - START_BATCH, 1).to(device)
    obs_eval_values = torch.empty((NUM_DAYS), BATCH_SIZE - START_BATCH, 1).to(device)
    yp_assim = torch.zeros([NUM_DAYS, 531, 1], dtype=torch.float32).detach()

    # Initialize variables
    numerator = 0


    # Initialize HBV models
    hbv = rnn.HBVMulTDET_DA6()

    # ==========================================================================
    # DATA ASSIMILATION LOOP
    # ==========================================================================

    print("\nStarting data assimilation...")
    print(f"Assimilation mode: {ASSIMILATION_MODE}")
    if ASSIMILATION_MODE in ['state_precip', 'precip_only']:
        print(f"Window adjustment position: a = {WINDOW_ADJUST_POSITION}")
    print("")

    for ind, j in enumerate(range(START_DAY, END_DAY)):
        # Show progress every 10 days
        if ind % 10 == 0 and OUTPUT_VERBOSITY == 'minimal':
            if TIME_MODE == 'calendar':
                current_date = index_to_date(j)
                print(f"\nProcessing day {j} ({current_date.strftime('%Y/%m/%d')})")
            else:
                print(f"\nProcessing day {j}")


        meltwater = meltwater_all[j - WINDOW_SIZE-1, :, :].float().cuda()
        sm = sm_all[j - WINDOW_SIZE-1, :, :].float().cuda()
        suz = suz_all[j - WINDOW_SIZE-1, :, :].float().cuda()
        slz = slz_all[j - WINDOW_SIZE-1, :, :].float().cuda()
        snowpack = snowpack_all[j - WINDOW_SIZE-1, :, :].float().cuda()

        end_day_1 = j + WINDOW_SIZE-1
        num = 0

        # Process windows
        for i in range(j, end_day_1, WINDOW_SIZE):
            # Prepare warm-up predictions
            yp_warm = yp_da[i - WINDOW_SIZE - 15:i - WINDOW_SIZE, :, 0]
            if numerator >= 15:
                yp_assim_array = yp_assim.detach().numpy()
                yp_warm = yp_assim_array[numerator - 15: numerator, :, 0]

            current_loss_checkpoint = None
            previous_loss_checkpoint = None


            # Initialize parameters and optimizer based on assimilation mode
            if ASSIMILATION_MODE == 'state_precip':
                # Both state variables and precipitation
                log_snowpack_adjusted = torch.nn.Parameter(torch.log(snowpack.clone().cuda()))
                log_meltwater_adjusted = torch.nn.Parameter(torch.log(meltwater.clone().cuda()))
                log_sm_adjusted = torch.nn.Parameter(torch.log(sm.clone().cuda()))
                log_suz_adjusted = torch.nn.Parameter(torch.log(suz.clone().cuda()))
                log_slz_adjusted = torch.nn.Parameter(torch.log(slz.clone().cuda()))
                k = nn.Parameter(torch.ones((BATCH_SIZE - START_BATCH)).cuda())

                optimizer = torch.optim.Adadelta(
                    [log_snowpack_adjusted, log_meltwater_adjusted, log_sm_adjusted,
                     log_suz_adjusted, log_slz_adjusted, k],
                    lr=LR1
                )

            elif ASSIMILATION_MODE == 'state_only':
                # Only state variables
                log_snowpack_adjusted = torch.nn.Parameter(torch.log(snowpack.clone().cuda()))
                log_meltwater_adjusted = torch.nn.Parameter(torch.log(meltwater.clone().cuda()))
                log_sm_adjusted = torch.nn.Parameter(torch.log(sm.clone().cuda()))
                log_suz_adjusted = torch.nn.Parameter(torch.log(suz.clone().cuda()))
                log_slz_adjusted = torch.nn.Parameter(torch.log(slz.clone().cuda()))

                optimizer = torch.optim.Adadelta(
                    [log_snowpack_adjusted, log_meltwater_adjusted, log_sm_adjusted,
                     log_suz_adjusted, log_slz_adjusted],
                    lr=lr1
                )

            else:  # precip_only
                # Only precipitation adjustment (k parameter)
                k = nn.Parameter(torch.ones((BATCH_SIZE - START_BATCH)).cuda())

                optimizer = torch.optim.Adadelta([k], lr=lr1)

            # Prepare window data
            window_forcing_p = torch.from_numpy(
                x_da_adjusted[i - WINDOW_SIZE:i + WINDOWS_UPPER, :, :]
            ).float().cuda()
            window_para_hbv = torch.from_numpy(
                para_hbv[i - WINDOW_SIZE:i + WINDOWS_UPPER, :, :, :]
            ).float().cuda()
            window_obs = torch.from_numpy(
                y_da[i - WINDOW_SIZE:i + WINDOWS_UPPER, :, :]
            ).float().cuda()
            window_ypretrain = torch.from_numpy(
                yp_da[i - WINDOW_SIZE:i + WINDOWS_UPPER, :, 0:1]
            ).float().cuda()
            window_yp_warmup = torch.from_numpy(yp_warm).float().cuda()

            # Initialize loss function
            loss_fun = crit.NSELossBatch3(np.nanstd(y_da[i - WINDOW_SIZE:i, :, :], axis=0))

            # Optimization loop
            loss = 0
            for iepoch in range(NUM_EPOCHS):
                if iepoch > 0 and torch.isnan(loss):
                    continue

                optimizer.zero_grad()

                # Prepare states based on mode
                if ASSIMILATION_MODE in ['state_precip', 'state_only']:
                    # Convert from log space
                    snowpack_adjusted = torch.exp(log_snowpack_adjusted)
                    meltwater_adjusted = torch.exp(log_meltwater_adjusted)
                    sm_adjusted = torch.exp(log_sm_adjusted)
                    suz_adjusted = torch.exp(log_suz_adjusted)
                    slz_adjusted = torch.exp(log_slz_adjusted)
                else:  # precip_only
                    # Use detached states (not optimized)
                    snowpack_adjusted = snowpack.detach()
                    meltwater_adjusted = meltwater.detach()
                    sm_adjusted = sm.detach()
                    suz_adjusted = suz.detach()
                    slz_adjusted = slz.detach()

                # Prepare forcing based on assimilation mode
                if ASSIMILATION_MODE in ['state_precip', 'precip_only']:
                    # Compute k factor and adjust forcing
                    k1 = k ** 4
                    window_forcing_p = torch.from_numpy(
                        x_da_adjusted[i - WINDOW_SIZE:i + WINDOWS_UPPER, :, :]
                    ).float().cuda()
                    window_forcing = window_forcing_p.clone()

                    # Apply k adjustment at the configurable position
                    a = WINDOW_ADJUST_POSITION
                    window_forcing[WINDOW_SIZE - a:WINDOW_SIZE - (a - 1), :, 0] = (
                        k1 * window_forcing_p[WINDOW_SIZE - a:WINDOW_SIZE - (a - 1), :, 0]
                    )
                else:  # state_only
                    # No forcing adjustment
                    window_forcing = window_forcing_p.clone()

                # Run HBV model with adjusted states/forcing
                out = hbv(
                    window_forcing, window_yp_warmup, window_para_hbv,
                    [snowpack_adjusted, meltwater_adjusted, sm_adjusted, suz_adjusted, slz_adjusted],
                    STA_IND, TD_REP, N_MUL, None, rtwts,
                    bufftime=0, outstate=False, routOpt=True,
                    comprout=False, dydrop=False
                )

                # Run HBV model without adjustment for comparison
                out_old = hbv(
                    window_forcing_p, window_yp_warmup, window_para_hbv,
                    [snowpack.detach(), meltwater.detach(), sm.detach(),
                     suz.detach(), slz.detach()],
                    STA_IND, TD_REP, N_MUL, None, rtwts,
                    bufftime=0, outstate=False, routOpt=True,
                    comprout=False, dydrop=False
                )

                window_yp_da = out[:, :, 0:1]
                background_yp = out_old[:, :, 0:1]

                # Compute losses
                loss = loss_fun(window_yp_da[0:WINDOW_SIZE, :, 0:1],
                               window_obs[0:WINDOW_SIZE, :, 0:1])
                loss_pretrain = loss_fun(background_yp[0:WINDOW_SIZE, :, 0:1],
                                       window_obs[0:WINDOW_SIZE, :, 0:1])

                # Simple output for optimization progress
                if OUTPUT_VERBOSITY == 'minimal':
                    print(f'Window {i}, iepoch {iepoch}: Loss {loss.item():.3f}')


                # Backward pass
                loss.backward(retain_graph=True)
                optimizer.step()

            # Update based on loss comparison
            if (loss < loss_pretrain) and not torch.isnan(loss):
                # Optimization was successful
                if ASSIMILATION_MODE in ['state_precip', 'state_only']:
                    # Get updated states
                    qsinit, snowpack_new, meltwater_new, sm_new, suz_new, slz_new = hbv(
                        window_forcing[0:WINDOW_SIZE - 1, :, :],
                        window_yp_warmup,
                        window_para_hbv[0:WINDOW_SIZE - 1, :, :, :],
                        [snowpack_adjusted, meltwater_adjusted, sm_adjusted, suz_adjusted, slz_adjusted],
                        STA_IND, TD_REP, N_MUL, None, rtwts,
                        bufftime=0, outstate=True, routOpt=False,
                        comprout=False, dydrop=False
                    )

                yp_assim[numerator:numerator + 1, :, :] = out[0:1, :, 0:1].detach().cpu()

                # Update forcing if precipitation was adjusted
                if ASSIMILATION_MODE in ['state_precip', 'precip_only']:
                    a = WINDOW_ADJUST_POSITION
                    x_da_adjusted[i - a:i - (a - 1), :, :] = (
                        window_forcing[WINDOW_SIZE - a:WINDOW_SIZE - (a - 1), :, :].detach().cpu().numpy()
                    )

                # Update states based on mode
                if ASSIMILATION_MODE in ['state_precip', 'state_only']:
                    state_variables_snowpack = snowpack_adjusted
                    state_variables_meltwater = meltwater_adjusted
                    state_variables_sm = sm_adjusted
                    state_variables_suz = suz_adjusted
                    state_variables_slz = slz_adjusted
                else:  # precip_only
                    state_variables_snowpack = snowpack
                    state_variables_meltwater = meltwater
                    state_variables_sm = sm
                    state_variables_suz = suz
                    state_variables_slz = slz
            else:
                # Keep original states and forcing
                state_variables_snowpack = snowpack
                state_variables_meltwater = meltwater
                state_variables_sm = sm
                state_variables_suz = suz
                state_variables_slz = slz

                qsinit, snowpack_new, meltwater_new, sm_new, suz_new, slz_new = hbv(
                    window_forcing_p[0:WINDOW_SIZE - 1, :, :],
                    window_yp_warmup,
                    window_para_hbv[0:WINDOW_SIZE - 1, :, :, :],
                    [snowpack.detach(), meltwater.detach(), sm.detach(),
                     suz.detach(), slz.detach()],
                    STA_IND, TD_REP, N_MUL, None, rtwts,
                    bufftime=0, outstate=True, routOpt=False,
                    comprout=False, dydrop=False
                )

                yp_assim[numerator:numerator + 1, :, :] = out_old[0:1, :, 0:1].detach().cpu()

            numerator += 1
            num += 1

            # Clear GPU cache
            torch.cuda.empty_cache()

            # Evaluation step
            if num == 1:
                out_eval = hbv(
                    torch.from_numpy(x_da_adjusted[i - WINDOW_SIZE:i + WINDOWS_UPPER, :, :]).float().cuda(),
                    window_yp_warmup, window_para_hbv,
                    [state_variables_snowpack, state_variables_meltwater,
                     state_variables_sm, state_variables_suz, state_variables_slz],
                    STA_IND, TD_REP, N_MUL, None, rtwts,
                    bufftime=0, outstate=False, routOpt=True,
                    comprout=False, dydrop=True
                )

                out_eval3 = out_eval[:WINDOW_SIZE + 1, :, :]
                obs_eval = window_obs[WINDOW_SIZE:WINDOW_SIZE + 1, :, :]

                out_eval_values[ind] = out_eval3[-1:, :, 0:1].detach()
                yp_w_da[ind] = window_ypretrain[WINDOW_SIZE:WINDOW_SIZE + 1, :, :].detach()
                obs_eval_values[ind] = obs_eval.detach()

                num = 0

    # ==========================================================================
    # EVALUATION AND SAVING
    # ==========================================================================

    print("\nComputing evaluation metrics...")

    # Convert tensors to numpy arrays
    out_eval_array = out_eval_values.cpu().numpy()
    obs_eval_array = obs_eval_values.cpu().numpy()
    yp_w_da_array = yp_w_da.cpu().numpy()

    # Squeeze dimensions
    out_eval_array_2d = np.squeeze(out_eval_array, axis=2)
    obs_eval_array_2d = np.squeeze(obs_eval_array, axis=2)
    out_eval_without_da_2d = np.squeeze(yp_w_da_array, axis=2)

    # Save results with mode suffix and date range
    mode_suffix = ASSIMILATION_MODE
    date_suffix = f"{START_DATE.strftime('%Y%m%d')}_{END_DATE.strftime('%Y%m%d')}" if TIME_MODE == 'calendar' else f"day{START_DAY}_{END_DAY}"

    print(f"Saving results with suffix: {mode_suffix}_{date_suffix}...")
    np.save(f"{SAVE_DIR}/out_eval_with_DA_{mode_suffix}_{date_suffix}.npy", out_eval_array_2d)
    np.save(f"{SAVE_DIR}/obs_eval_array_{mode_suffix}_{date_suffix}.npy", obs_eval_array_2d)
    np.save(f"{SAVE_DIR}/out_eval_without_DA_{mode_suffix}_{date_suffix}.npy", out_eval_without_da_2d)

    # Compute metrics after DA
    print(f"\nMetrics after data assimilation ({ASSIMILATION_MODE} mode):")
    eva_dict_after, data_box_after = compute_evaluation_metrics(out_eval_array, obs_eval_array)

    # Compute metrics before DA
    print("\nMetrics before data assimilation:")
    eva_dict_before, data_box_before = compute_evaluation_metrics(yp_w_da_array, obs_eval_array)

    # Print summary
    print("\n" + "="*60)
    print("ASSIMILATION SUMMARY")
    print("="*60)
    print(f"Mode: {ASSIMILATION_MODE}")
    if TIME_MODE == 'calendar':
        start_date_str = index_to_date(START_DAY).strftime('%Y/%m/%d')
        end_date_str = index_to_date(END_DAY - 1).strftime('%Y/%m/%d')  # -1 because range is exclusive
        print(f"Period: {start_date_str} to {end_date_str}")
    else:
        print(f"Period: Day {START_DAY} to Day {END_DAY - 1}")  # -1 because range is exclusive
    print(f"Days processed (DA_DAYS): {NUM_DAYS}")
    print("="*60)

    print("\nDone!")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
