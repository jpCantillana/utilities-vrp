import numpy as np
from random import randint

# Format relevant data in a class
class Customer():
    def __init__(self, id, xcoord, ycoord, demand, earliest_arrival, latest_departure, service_time):
        self.id = id
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.demand = demand
        self.earliest_arrival = earliest_arrival
        self.latest_departure = latest_departure
        self.service_time = service_time


class SolverDataInput():
    def __init__(self, instance_name):
        self.instance_name = instance_name
        self.vehicles      = 0
        self.capacity      = 0
        self.customers     = []

    def set_vehicles(self, n):
        self.vehicles = n
    
    def set_capacity(self, cap):
        self.capacity = cap
    
    def add_customer(self, id, xcoord, ycoord, demand, earliest_arrival, latest_departure, service_time):
        self.customers.append(Customer(id, xcoord, ycoord, demand, earliest_arrival, latest_departure, service_time))

space_scenarios = 10

# spatial random var generation
pi_vals = (np.random.uniform(low=0.0,high=1.0,size=space_scenarios)).tolist()
mu_lat_vals = (np.random.uniform(low=5.0,high=25.0,size=space_scenarios)).tolist()
mu_lon_vals = (np.random.uniform(low=5.0,high=25.0,size=space_scenarios)).tolist()
sigma_11_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
sigma_12_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
sigma_21_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
sigma_22_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
mu_demand_vals = (np.random.uniform(low=0.1, high=10.0, size=space_scenarios)).tolist()

s_demand_alternatives = [[1, mu_demand_vals[i]/3] for i in range(space_scenarios)]
tw_strategies = ['B2B', 'B2C', 'mix']


scenarios = {}
for i in range(space_scenarios):
    scenarios[i] = {
        "pi": pi_vals[i],
        "mu_lat": mu_lat_vals[i],
        "mu_lon": mu_lon_vals[i],
        "s11": sigma_11_vals[i],
        "s12": sigma_12_vals[i],
        "s21": sigma_21_vals[i],
        "s22": sigma_22_vals[i],
        "mu_demand": mu_demand_vals[i],
        "laterals": [
            (
                {
                "s_demand": s_demand_alternatives[i][j],
                "tw_strategy": tw_strategies[k]
                }
            ) for j in range(2) for k in range(3)
        ]
    }
print(scenarios)

realizations = 3
for index, main_scenario in scenarios.items():
    #locations
    mean = np.array([main_scenario['mu_lat'], main_scenario['mu_lon']])
    covariance = np.array([[main_scenario["s11"], main_scenario["s12"]],[main_scenario["s21"],main_scenario["s22"]]])
    latlon = np.random.multivariate_normal(mean, covariance, size=realizations)
    for lateral in main_scenario["laterals"]:
        if lateral["tw_strategy"] == 'B2B':
            #roll over realizations
                #check if it's going to be constrainted or not
                #if constrainted
                    #then choose if early or late
                    #if early, determine earliest arrival y latest departure based on pre-made ones
            pass
        elif lateral["tw_strategy"] == 'B2C':
            pass
        else:
            pass