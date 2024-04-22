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

class InstanceCreator():
    def __init__(self, scenario_size=100):
        self.scenarios = {}
        self.realisations = {}
        self.scenario_size = scenario_size
        self.n_clusters = 1

    def reset_scenarios(self):
        self.scenarios = {}
    
    def add_scenario(self, name, realisations, cust_dist, depot_location, n_clusters):
        self.scenarios[name] = (realisations, cust_dist, depot_location, n_clusters)
        self.realisations[name] = []
    
    def add_realisations(self, name, customer_locations):
        self.realisations[name].append({"locations": customer_locations})
    
    def set_balancing(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    
    def set_normal_parameters(self, mu1, mu2, sigma11, sigma12, sigma21, sigma22):
        self.mu1 = mu1
        self.mu2 = mu2
        self.sigma11= sigma11
        self.sigma12 = sigma12
        self.sigma21 = sigma21
        self.sigma22 = sigma22
    
    def set_uniform_parameters(self, lb, ub):
        self.lb = lb
        self.ub = ub
    
    def set_auxiliar_uniform_clusters(self, bounds_list):
        self.uniform_clusters = [(self.lb, self.ub)] + bounds_list

    def set_auxiliar_normal_clusters(self, parameters_list):
        self.normal_clusters = [(self.mu1, self.mu2, self.sigma11, self.sigma12, self.sigma21, self.sigma22)] + parameters_list
    
    def double_uniform_generation(self):
        customer_locations = []
        if self.n_clusters > 1:
            cluster_map = {j:int(j//(self.scenario_size//self.n_clusters)) for j in range(self.scenario_size)}
        for i in range(self.scenario_size):
            if self.n_clusters > 1:
                lb, ub = self.uniform_clusters[cluster_map[i]]
            else:
                lb, ub = self.lb, self.ub
            lat = randint(lb, ub)
            lon = randint(lb, ub)
            customer_locations.append((lat, lon))
        return customer_locations
    
    def bivariate_normal_generation(self):
        customer_locations = []
        if self.n_clusters > 1:
            cluster_map = {j:int(j//(self.scenario_size//self.n_clusters)) for j in range(self.scenario_size)}
        mu = np.array([self.mu1,self.mu2])
        covariance = np.array([[self.sigma11**2, self.sigma12**2],[self.sigma21**2, self.sigma22**2]])
        for i in range(self.scenario_size):
            if self.n_clusters > 1:
                mu1, mu2, sigma11, sigma12, sigma21, sigma22 = self.normal_clusters[cluster_map[i]]
                mu = np.array([mu1,mu2])
                covariance = np.array([[sigma11**2, sigma12**2],[sigma21**2, sigma22**2]])
            latlon = np.random.multivariate_normal(mu, covariance, size=1).tolist()[0]
            latlon = [max(int(j//1), 0) for j in latlon]
            customer_locations.append(tuple(latlon))
        return customer_locations
    
    def double_normal_generation(self):
        customer_locations = []
        if self.n_clusters > 1:
            cluster_map = {j:int(j//(self.scenario_size//self.n_clusters)) for j in range(self.scenario_size)}
        for i in range(self.scenario_size):
            if self.n_clusters > 1:
                mu1, mu2, sigma11, _, _, sigma22 = self.normal_clusters[cluster_map[i]]
            else:
                mu1, mu2, sigma11, sigma22 = self.mu1, self.mu2, self.sigma11, self.sigma22
            lat = max(int(np.random.normal(mu1, sigma11, size=1).tolist()[0]//1),0)
            lon = max(int(np.random.normal(mu2, sigma22, size=1).tolist()[0]//1),0)
            customer_locations.append((lat, lon))
        return customer_locations

    def location_algo(self):
        for scenario, params in self.scenarios.items():
            n_realisations, customer_dist, depot_location, n_clusters = params
            if n_clusters > 0:
                self.n_clusters = n_clusters + 1
            if customer_dist == "double_uniform":
                for i in range(n_realisations):
                    latlon = self.double_uniform_generation()
                    self.add_realisations(scenario, latlon)
            elif customer_dist == "bivariate_normal":
                for i in range(n_realisations):
                    latlon = self.bivariate_normal_generation()
                    self.add_realisations(scenario, latlon)
            elif customer_dist == "double_normal":
                for i in range(n_realisations):
                    latlon = self.double_normal_generation()
                    self.add_realisations(scenario, latlon)
            elif customer_dist == "linear_combination":
                for i in range(n_realisations):
                    customer_locations = []
                    latlon1 = self.double_uniform_generation()
                    latlon2 = self.bivariate_normal_generation()
                    latlon3 = self.double_normal_generation()
                    for j in range(self.scenario_size):
                        lat = self.alpha * latlon1[j][0] + self.beta * latlon2[j][0] + (1 - self.alpha - self.beta) * latlon3[j][0]
                        lon = self.alpha * latlon1[j][1] + self.beta * latlon2[j][1] + (1 - self.alpha - self.beta) * latlon3[j][1]
                        customer_locations.append((int(lat//1), int(lon//1)))
                    self.add_realisations(scenario, customer_locations)
            self.n_clusters = 1

instance_object = InstanceCreator()
instance_object.add_scenario("double_uniform_test", 1, "double_uniform", 0, 1)
instance_object.add_scenario("bivariate_normal_test", 1, "bivariate_normal", 0, 1)
instance_object.add_scenario("double_normal_test", 1, "double_normal", 0, 0)
instance_object.add_scenario("linear_combination_test", 1, "linear_combination", 0, 0)
instance_object.set_balancing(0.4,0.5)
instance_object.set_uniform_parameters(0, 200)
instance_object.set_normal_parameters(100, 80, 10,10,2,4)
instance_object.set_auxiliar_uniform_clusters([(200, 250)])
instance_object.set_auxiliar_normal_clusters([(300,500,10,20,20,10)])
instance_object.location_algo()


# space_scenarios = 10

# # spatial random var generation
# pi_vals = (np.random.uniform(low=0.0,high=1.0,size=space_scenarios)).tolist()
# mu_lat_vals = (np.random.uniform(low=5.0,high=25.0,size=space_scenarios)).tolist()
# mu_lon_vals = (np.random.uniform(low=5.0,high=25.0,size=space_scenarios)).tolist()
# sigma_11_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
# sigma_12_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
# sigma_21_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
# sigma_22_vals = (np.random.uniform(low=3.0,high=10.0,size=space_scenarios)).tolist()
# mu_demand_vals = (np.random.uniform(low=0.1, high=10.0, size=space_scenarios)).tolist()

# s_demand_alternatives = [[1, mu_demand_vals[i]/3] for i in range(space_scenarios)]
# tw_strategies = ['B2B', 'B2C', 'mix']


# scenarios = {}
# for i in range(space_scenarios):
#     scenarios[i] = {
#         "pi": pi_vals[i],
#         "mu_lat": mu_lat_vals[i],
#         "mu_lon": mu_lon_vals[i],
#         "s11": sigma_11_vals[i],
#         "s12": sigma_12_vals[i],
#         "s21": sigma_21_vals[i],
#         "s22": sigma_22_vals[i],
#         "mu_demand": mu_demand_vals[i],
#         "laterals": [
#             (
#                 {
#                 "s_demand": s_demand_alternatives[i][j],
#                 "tw_strategy": tw_strategies[k]
#                 }
#             ) for j in range(2) for k in range(3)
#         ]
#     }
# print(scenarios)

# realizations = 3
# for index, main_scenario in scenarios.items():
#     #locations
#     mean = np.array([main_scenario['mu_lat'], main_scenario['mu_lon']])
#     covariance = np.array([[main_scenario["s11"], main_scenario["s12"]],[main_scenario["s21"],main_scenario["s22"]]])
#     latlon = np.random.multivariate_normal(mean, covariance, size=realizations)
#     for lateral in main_scenario["laterals"]:
#         if lateral["tw_strategy"] == 'B2B':
#             #roll over realizations
#                 #check if it's going to be constrainted or not
#                 #if constrainted
#                     #then choose if early or late
#                     #if early, determine earliest arrival y latest departure based on pre-made ones
#             pass
#         elif lateral["tw_strategy"] == 'B2C':
#             pass
#         else:
#             pass