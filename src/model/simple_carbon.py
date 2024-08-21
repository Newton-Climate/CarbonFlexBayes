import numpy as np
from scipy.integrate import odeint
import pandas as pd

class DiscreteCarbonModel:
    def __init__(self, num_layers=1, num_pools=3, turnover_times=None, fractionation_rates=None):
        self.num_layers = num_layers
        self.num_pools = num_pools
        
        # Initialize turnover times with defaults (5, 50, 500 years) or user-provided values
        default_turnover = np.array([[5, 50, 500] for _ in range(num_layers)])
        self.turnover_times = np.array(turnover_times if turnover_times is not None else default_turnover)
        
        # Initialize default fractionation rates or use user-provided values
        default_fractionation_rates = {'C13': 0.975, 'C14': 0.975**2}
        self.fractionation_rates = fractionation_rates if fractionation_rates is not None else default_fractionation_rates
        
        # Initialize transfer rates with specified values for fast to slow and slow to passive pools
        self.transfer_rates = self.initialize_transfer_rates()


    def initialize_transfer_rates(self):
        """Initialize transfer rates with specified values."""
        transfer_rates = np.zeros((self.num_layers, self.num_pools, self.num_layers, self.num_pools))
        
        # Set the transfer rates from fast to slow and slow to passive pools
        for layer in range(self.num_layers):
            transfer_rates[layer, 0, layer, 1] = 0.02  # 2% from fast to slow
            transfer_rates[layer, 1, layer, 2] = 0.0002  # 0.02% from slow to passive
        
        return transfer_rates


    def construct_transfer_matrix(self):
        """Construct the transfer matrix A using initialized transfer rates."""
        block_size = self.num_pools * 3
        A = np.zeros((self.num_layers * block_size, self.num_layers * block_size))

        for layer in range(self.num_layers):
            for pool in range(self.num_pools):
                idx = layer * block_size + pool * 3
                A[idx, idx] = -1.0 / self.turnover_times[layer, pool]
                A[idx+1, idx+1] = A[idx, idx] * self.fractionation_rates['C13']
                A[idx+2, idx+2] = A[idx, idx] * self.fractionation_rates['C14']

                for other_layer in range(self.num_layers):
                    for other_pool in range(self.num_pools):
                        other_idx = other_layer * block_size + other_pool * 3
                        if layer == other_layer and pool != other_pool:
                            rate = self.transfer_rates[layer, pool, other_layer, other_pool]
                            A[idx:idx+3, other_idx:other_idx+3] = np.array([rate, rate * self.fractionation_rates['C13'], rate * self.fractionation_rates['C14']])

        return A


    def construct_forcing_vector(self, gpp, nee, reco):
        """Construct the forcing vector F using inputs."""
        block_size = self.num_pools * 3
        F = np.zeros(self.num_layers * block_size)

        for layer in range(self.num_layers):
            for pool in range(self.num_pools):
                pool_idx = layer * block_size + pool * 3
                if pool == 0:  # Fast pool receives GPP
                    F[pool_idx] = gpp
                    F[pool_idx + 1] = gpp * self.fractionation_rates['C13']
                    F[pool_idx + 2] = gpp * self.fractionation_rates['C14']
                elif pool == 1:  # Slow pool receives NEE
                    F[pool_idx] = nee
                    F[pool_idx + 1] = nee * self.fractionation_rates['C13']
                    F[pool_idx + 2] = nee * self.fractionation_rates['C14']
                else:  # Passive pool, no direct input
                    F[pool_idx:pool_idx + 3] = 0

        return F

    def dynamics(self, state, t, gpp, nee, reco):
        """Compute the dynamics of the system."""
        A = self.construct_transfer_matrix()
        F = self.construct_forcing_vector(gpp, nee, reco)
        self.A = A
        self.F = F
        self.gpp_fluxes.append(F[0])
        self.nee_fluxes.append(F[1])
        self.reco_fluxes.append(F[0] + F[1])
        decomposition =  np.dot(A, state)
        state_change =         decomposition + F

        self.reco_fluxes.append(np.sum(decomposition[0::3]))

        return state_change

    def run_simulation(self, initial_state, time_points, gpp, nee, reco):
        """Run the simulation over a specified time period."""
        # Pack the additional arguments (gpp, nee, reco) into a tuple
        args = (gpp, nee, reco)

        # create list to store flux time-series to compare with obs 
        self.gpp_fluxes = []
        self.nee_fluxes = []
        self.reco_fluxes = []
        
        # Use odeint to integrate the system of differential equations
        result = odeint(self.dynamics, initial_state, time_points, args=args)
        
        return result


# Initialize the model
model = DiscreteCarbonModel()

# Define initial state, time points, and rates
initial_state = np.ones(9) * 100  # Example initial state
initial_state[0] = 1.0e3
initial_state[3] = 3.0e3
initial_state[6] = 5.0e3

time_points = np.linspace(0, 10, 101)  # Time points over 10 years
gpp = 2.0e3  # Example GPP rate
nee = -200  # Example NEE rate
reco = 0.50 * gpp  # Example Reco rate

# read the flux-net data file 
filename = '/Users/newtonnguyen/Documents/projects/RadioCarbon/data/flux-net/AMF_US-Var_FLUXNET_FULLSET_2000-2021_3-5/AMF_US-Var_FLUXNET_FULLSET_HH_2000-2021_3-5.csv'
ds = pd.read_csv(filename)

nee = ds.NEE_VUT_REF.values
reco = ds.RECO_SR.values
gpp = nee - reco
print(gpp)


# Run the simulation
results = model.run_simulation(initial_state, time_points, gpp, nee, reco)

print(results)
