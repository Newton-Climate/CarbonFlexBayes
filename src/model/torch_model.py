import torch
import pandas as pd
#from torch import torchdiffeq
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from iosacal import R, combine, iplot

from scipy.interpolate import interp1d


import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
from rpy2.robjects import pandas2ri
import iosacal
from iosacal.core import calibrate

# Import the rintcal package
rintcal = importr('rintcal')
base = importr('base')

# Load the post-bomb calibration curve
postbomb_curve = rintcal.ccurve("sh1-2_monthly")

# Get the column names
column_names = postbomb_curve.colnames
print(f"Column names: {column_names}")

# Convert the R data.frame to a pandas DataFrame
pandas2ri.activate()
postbomb_curve_df = pandas2ri.rpy2py(postbomb_curve)
print("\nConverted pandas DataFrame:")
print(postbomb_curve_df.head())

def calculate_age(modern_fraction=None, delta_14C=None, sigma_m=0.01, sigma_t=0.01):
    """
    Calculate the radiocarbon age given either a modern fraction or Δ14C.
    
    Parameters:
    modern_fraction (float): The modern fraction (Fm) of the sample.
    delta_14C (float): The Δ14C (Delta14C) value of the sample in per mil.
    sigma_m (float): The uncertainty in the modern fraction measurement.
    sigma_t (float): The uncertainty in the true fraction measurement.
    
    Returns:
    float: The radiocarbon age in years BP.
    """
    if modern_fraction is not None:
        if modern_fraction > 1:
            # Post-bomb carbon
            # Calculate the calendar year AD from modern fraction
            post_bomb_match = postbomb_curve_df.loc[np.round(postbomb_curve_df['V2'], 2) == np.round(modern_fraction, 2)]
            if not post_bomb_match.empty:
                age = post_bomb_match['V1'].iloc[0]
            else:
                raise ValueError(f"No post-bomb calibration data available for modern fraction: {modern_fraction}")
        else:
            # Pre-bomb carbon
            cal_age = calibrate(f_m=modern_fraction, sigma_m=sigma_m, f_t=modern_fraction, sigma_t=sigma_t)
            age = float(cal_age)
    elif delta_14C is not None:
        modern_fraction = (delta_14C / 1000) + 1
        if modern_fraction > 1:
            # Post-bomb carbon
            # Calculate the calendar year AD from modern fraction
            post_bomb_match = postbomb_curve_df.loc[np.round(postbomb_curve_df['V2'], 2) == np.round(modern_fraction, 2)]
            if not post_bomb_match.empty:
                age = post_bomb_match['V1'].iloc[0]
            else:
                raise ValueError(f"No post-bomb calibration data available for modern fraction: {modern_fraction}")
        else:
            # Pre-bomb carbon
            cal_age = calibrate(f_m=modern_fraction, sigma_m=sigma_m, f_t=modern_fraction, sigma_t=sigma_t)
            age = float(cal_age)
    else:
        raise ValueError("Either modern_fraction or delta_14C must be provided.")
    
    return age

def calculate_modern_fraction_or_delta_14C(age, return_type="modern_fraction", sigma_m=0.01, sigma_t=0.01):
    """
    Calculate the modern fraction or Δ14C given a radiocarbon age.
    
    Parameters:
    age (float): The radiocarbon age in years BP.
    return_type (str): The type of value to return, either "modern_fraction" or "delta_14C".
    sigma_m (float): The uncertainty in the modern fraction measurement.
    sigma_t (float): The uncertainty in the true fraction measurement.
    
    Returns:
    float: The modern fraction or Δ14C value.
    """
    if age < 0:
        # Post-bomb carbon
        post_bomb_match = postbomb_curve_df.loc[np.round(postbomb_curve_df['V1'], 4) == np.round(age, 4)]
        if not post_bomb_match.empty:
            modern_fraction = post_bomb_match['V2'].iloc[0]
        else:
            raise ValueError(f"No post-bomb calibration data available for age: {age}")
    else:
        # Pre-bomb carbon
        cal_age = calibrate(f_m=age, sigma_m=sigma_m, f_t=age, sigma_t=sigma_t)
        modern_fraction = float(cal_age)

    if return_type == "modern_fraction":
        return modern_fraction
    elif return_type == "delta_14C":
        delta_14C = (modern_fraction - 1) * 1000
        return delta_14C
    else:
        raise ValueError("return_type must be either 'modern_fraction' or 'delta_14C'")

# Example usage
modern_fraction_example = 0.85  # Example modern fraction
delta_14C_example = -150  # Example Δ14C value in per mil
age_example = 4000  # Example age in years BP
age_postbomb_example = -69.3750  # Example post-bomb age in years BP (1980.625 AD)

# Calculate age using modern fraction
age_from_modern_fraction = calculate_age(modern_fraction=modern_fraction_example)
print(f"Radiocarbon age from modern fraction: {age_from_modern_fraction} years BP")

# Calculate age using Δ14C
age_from_delta_14C = calculate_age(delta_14C=delta_14C_example)
print(f"Radiocarbon age from Δ14C: {age_from_delta_14C} years BP")

# Calculate modern fraction from age
modern_fraction_from_age = calculate_modern_fraction_or_delta_14C(age=age_example, return_type="modern_fraction")
print(f"Modern fraction from age: {modern_fraction_from_age}")

# Calculate Δ14C from age
delta_14C_from_age = calculate_modern_fraction_or_delta_14C(age=age_example, return_type="delta_14C")
print(f"Δ14C from age: {delta_14C_from_age} per mil")

# Calculate modern fraction from post-bomb age
modern_fraction_from_postbomb_age = calculate_modern_fraction_or_delta_14C(age=age_postbomb_example, return_type="modern_fraction")
print(f"Modern fraction from post-bomb age: {modern_fraction_from_postbomb_age}")

# Calculate Δ14C from post-bomb age
delta_14C_from_postbomb_age = calculate_modern_fraction_or_delta_14C(age=age_postbomb_example, return_type="delta_14C")
print(f"Δ14C from post-bomb age: {delta_14C_from_postbomb_age} per mil")

def preprocess_data(dataframe, dtype=torch.float32):
    """
    Converts necessary columns from a pandas DataFrame into PyTorch tensors.

    Args:
    - dataframe (pd.DataFrame): The DataFrame containing the model's input data.

    Returns:
    - dict: A dictionary containing the tensors needed for the model.
    """
    # dataframe['TIMESTAMP_START'] = pd.to_datetime(dataframe['TIMESTAMP_START'])

    dataframe['TIMESTAMP_START'] = dataframe['TIMESTAMP_START'].astype(str)
    dataframe['TIMESTAMP_START'] = pd.to_datetime(dataframe['TIMESTAMP_START'])
    #dataframe.set_index('TIMESTAMP_START', inplace=True, drop=True)
    
    dataframe.sort_values('TIMESTAMP_START', inplace=True)
    dataframe = dataframe.drop_duplicates(subset='TIMESTAMP_START', keep='first')
    dataframe.set_index('TIMESTAMP_START', inplace=True)
    
    

    # Convert timestamps to numeric time (seconds from start)
    time_points = (dataframe.index - dataframe.index[0]).total_seconds()
    time_hours = time_points / 3600  # Convert seconds to hours
    
    # Convert series to tensors
    time_hours_tensor = torch.tensor(time_hours.values, dtype=dtype)

    if not all(time_hours[1:] > time_hours[:-1]):
        raise ValueError("Time hours are not strictly increasing after preprocessing.")
    
    gpp_tensor = torch.tensor(dataframe['GPP_DT_VUT_REF'].values * 0.5, dtype=dtype)
    nee_tensor = torch.tensor(dataframe['NEE_VUT_REF'].values, dtype=dtype)
    reco_tensor = -1.0 * torch.tensor(dataframe['RECO_NT_VUT_REF'].values, dtype=dtype)
    
    return {
        'time_hours': time_hours_tensor,
        'gpp': gpp_tensor,
        'nee': nee_tensor,
        'reco': reco_tensor
    }

def rate2hourly(annual_rate_percent):
    annual_rate_decimal = annual_rate_percent / 100.0
    hourly_rate_decimal = (1 + annual_rate_decimal) ** (1 / (24 * 365.25)) - 1
    return hourly_rate_decimal


class LinearInterpolation(nn.Module):
    def __init__(self, times, values):
        super(LinearInterpolation, self).__init__()
        self.times = times
        self.values = values

    def forward(self, t):
        # Find indices of the closest time points
        indices = torch.searchsorted(self.times, t, right=True)
        indices = indices.clamp(min=1, max=len(self.times) - 1)
        t0 = self.times[indices - 1]
        t1 = self.times[indices]
        v0 = self.values[indices - 1]
        v1 = self.values[indices]

        # Linear interpolation
        fraction = (t - t0) / (t1 - t0)
        result = v0 + fraction * (v1 - v0)
        # print(t)
        return result



class DiscreteCarbonModel(torch.nn.Module):
    def __init__(self, num_layers=1, num_pools=3, turnover_times=None, fractionation_rates=None):
        super(DiscreteCarbonModel, self).__init__()
        self.num_layers = num_layers
        self.num_pools = num_pools
        default_turnover = torch.tensor([[3.0, 30.0, 3000.0] for _ in range(num_layers)]).float() * 24 * 365.25
        
        self.turnover_times = torch.tensor(turnover_times if turnover_times is not None else default_turnover).float()
        self.fractionation_rates = fractionation_rates if fractionation_rates is not None else {'C13': 0.975, 'C14': 0.975**2}
        
        self.transfer_rates = self.initialize_transfer_rates()

    def initialize_transfer_rates(self):
        transfer_rates = torch.zeros((self.num_layers, self.num_pools, self.num_layers, self.num_pools))
        transfer_rates[:, 0, :, 1] = 0.02
        transfer_rates[:, 1, :, 2] = 0.002
        return transfer_rates


    def construct_transfer_matrix(self):
        """Construct the transfer matrix A using initialized transfer rates."""
        block_size = self.num_pools * 3
        A = torch.zeros((self.num_layers * block_size, self.num_layers * block_size), dtype=torch.float32)
        daily_lambda = torch.log(torch.tensor(2.0)) / (24 * 365.25 * 5730)  # C14 decay rate from years to hours

        for layer in range(self.num_layers):
            for pool in range(self.num_pools):
                idx = layer * block_size + pool * 3
                A[idx, idx] = -1.0 / self.turnover_times[layer, pool]
                A[idx+1, idx+1] = A[idx, idx] * self.fractionation_rates['C13']
                A[idx+2, idx+2] = A[idx, idx] * self.fractionation_rates['C14'] - daily_lambda

                for other_layer in range(self.num_layers):
                    for other_pool in range(self.num_pools):
                        other_idx = other_layer * block_size + other_pool * 3
                        if layer == other_layer and pool != other_pool:
                            rate = rate2hourly(self.transfer_rates[layer, pool, other_layer, other_pool])
                            rate_tensor = torch.tensor([rate, rate * self.fractionation_rates['C13'], rate * self.fractionation_rates['C14']], dtype=torch.float32)
                            A[idx:idx+3, other_idx:other_idx+3] = rate_tensor
        self.A = A
        return A
    


    def construct_forcing_vector(self, t):
        block_size = self.num_pools * 3
        F = torch.zeros(self.num_layers * block_size)
        mortality = 0.4
        # Values interpolated at time t using PyTorch operations
        
        # Interpolate at time t
        gpp = self.gpp_interp.forward(t)
        nee = self.nee_interp.forward(t)
        reco = self.reco_interp.forward(t)
        fast_debris = 0.7 * mortality * gpp
        slow_debris = 0.3 * mortality * gpp

        for layer in range(self.num_layers):
            idx = layer * block_size
            F[idx] = fast_debris
            F[idx + 1] = fast_debris * self.fractionation_rates['C13']
            F[idx + 2] = fast_debris * self.fractionation_rates['C14']
            F[idx + 3] = slow_debris
            F[idx + 4] = slow_debris * self.fractionation_rates['C13']
            F[idx + 5] = slow_debris * self.fractionation_rates['C14']
        return F

    def forward(self, t, state):
        A = self.construct_transfer_matrix()
        F = self.construct_forcing_vector(t)
        return torch.matmul(A, state) + F

    def run_simulation(self, initial_state, forcings):
        # Convert time_points from DateTime to Float
        # time_hours = (time_points - time_points[0]).total_seconds() / 24

        time_hours = forcings['time_hours'].clone().detach()
        if not (time_hours[1:] > time_hours[:-1]).all():
            print(time_hours)
            raise ValueError("time_hours tensor must be strictly increasing.")
        
        gpp_series = forcings['gpp']
        nee_series = forcings['nee']
        reco_series = forcings['reco']

        # Create tensors from data series
        self.set_time_series(time_hours, gpp_series, nee_series, reco_series)
        
        # Run ODE solver
        result = odeint(self, initial_state, time_hours)
        return result


    def set_time_series(self, time_hours, gpp_series, nee_series, reco_series):
        # initialize the interpolators 
        self.gpp_interp = LinearInterpolation(time_hours, gpp_series)
        self.nee_interp = LinearInterpolation(time_hours, nee_series)
        self.reco_interp = LinearInterpolation(time_hours, reco_series)




# Read the CSV file
filename = '/Users/newtonnguyen/Documents/projects/RadioCarbon/data/flux-net/AMF_US-Ha1_FLUXNET_FULLSET_1991-2020_3-5/AMF_US-Ha1_FLUXNET_FULLSET_HR_1991-2020_3-5.csv'
ds = pd.read_csv(filename)


forcings = preprocess_data(ds)


# Define initial state
initial_state = torch.ones(9) * 100  # Example initial state
initial_state[0:2] = 11e3
initial_state[3:5] = 3.0e3
initial_state[6:8] = 7.0e3


# Initialize the model
model = DiscreteCarbonModel()

# Run the simulation
spin_up = model.run_simulation(initial_state, forcings)
steady_state = spin_up[-1,:]

# Display steady-state values
print(f"steady-state values are {steady_state[::3]}")

# Run the simulation from steady state for result analysis
modern_14C_concentration = 226  # Modern level in Bq/kg
fraction_modern = [1.2, 0, 0] # radiocarbon pool

# Radiocarbon pools are the 3rd, 6th, and 9th in your indexing (2, 5, 8 in 0-based indexing)
# Adjust the initial state for radiocarbon concentrations
initial_state[2] = initial_state[2] * modern_14C_concentration * fraction_modern[0]
initial_state[5] = initial_state[5] * modern_14C_concentration * fraction_modern[1]
initial_state[8] = initial_state[8] * modern_14C_concentration * fraction_modern[2]


results = model.run_simulation(steady_state, forcings)

steady_state = results[-1,:]

# Display steady-state values
print(f"steady-state values are {steady_state[::3]}")

### Calculate the rate of change of the system at each time step
# Ensure that model.A is a torch tensor and results are also tensors
rate_of_change = [torch.matmul(model.A, results[i, :]) for i in range(len(forcings['time_hours']))]

# Initialize an empty list to store the total respiration flux at each time step
total_respiration_flux = []

# For each rate of change vector, sum the elements corresponding to total respiration
for rate in rate_of_change:
    # Sum every third element starting from the first one to get total respiration
    # torch.narrow can be used to slice out every third element from the tensor
    total_flux = torch.sum(rate[::3])  # Slices out every third element
    total_respiration_flux.append(total_flux)

# Convert list to a PyTorch tensor
total_respiration_flux = torch.stack(total_respiration_flux)

# Assuming 'reco' is a tensor containing respiration data to which you want to compare
# Make sure 'reco' is of the same length and properly aligned with 'total_respiration_flux'
if 'reco' in forcings:
    reco = forcings['reco']
    # Safety check to ensure alignment in dimensions
    if reco.shape[0] != total_respiration_flux.shape[0]:
        raise ValueError("Dimension mismatch between recorded respiration data and calculated flux.")
    respiration_ratio = total_respiration_flux / reco
    print(respiration_ratio)
else:
    print("Respiration data 'reco' not found in forcings.")



import matplotlib.pyplot as plt
t = forcings['time_hours']
plt.plot(t, total_respiration_flux, color='black', label='model respiration')
plt.plot(t, reco, color='orange', label='RECO')
plt.plot(t, forcings['gpp'], color='green', label='GPP')
plt.xlabel('hours')
plt.ylabel(r'g/m^2/s')
plt.xlim(0, 365*24)
plt.savefig('diagnostics.png')

