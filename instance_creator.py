import numpy as np
from random import randint
from math import sqrt
from sys import float_info
import pickle

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
    
    def add_scenario(self, name, realisations, cust_dist, depot_location, n_clusters, demand_dist, cap, fleet, tw, tw_dist, alpha=0, beta=0):
        self.scenarios[name] = (realisations, cust_dist, depot_location, n_clusters, demand_dist, cap, fleet, tw, tw_dist, alpha, beta)
        self.realisations[name] = {"locations": [], "depots": [], "demands": [], "capacities": [], "service_times": [], "time_windows": [], "fleets": []}
    
    def add_realisations(self, name, customer_locations):
        self.realisations[name]["locations"].append(customer_locations)
    
    def add_depot(self, name, depot_location):
        self.realisations[name]["depots"].append(depot_location)
    
    def add_demand(self, name, demands):
        self.realisations[name]["demands"].append(demands)
    
    def add_capacity(self, name, cap):
        self.realisations[name]["capacities"].append(cap)

    def add_service_time(self, name, service_times):
        self.realisations[name]["service_times"].append(service_times)

    def add_time_window(self, name, time_windows):
        self.realisations[name]["time_windows"].append(time_windows)

    def add_fleet(self, name, fleet):
        self.realisations[name]["fleets"].append(fleet)

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
        n_sc = len(self.scenarios)
        cnt = 0
        for scenario, params in self.scenarios.items():
            print("Creating scenario "+str(cnt)+" of "+str(n_sc))
            n_realisations, customer_dist, depot_location, n_clusters, _, _, _, _, _, a, b = params
            if depot_location == "central":
                self.add_depot(scenario, (int((self.ub-self.lb)/2//1), int((self.ub-self.lb)/2//1)))
            elif depot_location == "annular":
                self.add_depot(scenario, (int((self.ub-self.lb)/4//1), int((self.ub-self.lb)/4//1)))
            else:
                self.add_depot(scenario, (int((self.ub-self.lb)/10//1), int((self.ub-self.lb)/10//1)))
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
                self.alpha = a
                self.beta = b
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
            cnt += 1
    
    def euclidean_dist(self, lat1, lon1, lat2, lon2):
        return sqrt((lat1-lat2)**2 + (lon1 - lon2)**2)
    
    def get_max_min_distance(self, name):
        depot_lat, depot_lon = self.realisations[name]["depots"][0]
        max_dist = -1
        min_dist = float_info.max
        for latlon in self.realisations[name]["locations"][0]:
            lat, lon = latlon
            distance = self.euclidean_dist(depot_lat, depot_lon, lat, lon)
            if distance > max_dist:
                max_dist = distance
            if distance < min_dist:
                min_dist = distance
        return max_dist, min_dist
    
    def add_demands(self):
        for scenario, params in self.scenarios.items():
            n_realisations, _, _, _, demand_dist, _, _, _, _, _, _ = params
            for j in range(n_realisations):
                if demand_dist == "constant":
                    demands_list = []
                    service_times = []
                    for i in range(self.scenario_size):
                        demands_list.append(5)
                        service_times.append(5)
                    self.add_demand(scenario, demands_list)
                    self.add_service_time(scenario, service_times)
                elif demand_dist == "uniform":
                    demands_list = []
                    service_times = []
                    for i in range(self.scenario_size):
                        demands_list.append(randint(1,10))
                        if demands_list[-1] > 9:
                            service_times.append(10)
                        else:
                            service_times.append(5)
                    self.add_demand(scenario, demands_list)
                    self.add_service_time(scenario, service_times)
                elif demand_dist == "normal":
                    demands_list = []
                    service_times = []
                    for i in range(self.scenario_size):
                        demands_list.append(max(int(np.random.normal(5, 2, size=1).tolist()[0]//1),0))
                        if demands_list[-1] > 9:
                            service_times.append(10)
                        else:
                            service_times.append(5)
                    self.add_demand(scenario, demands_list)
                    self.add_service_time(scenario, service_times)
                elif demand_dist == "poisson":
                    demands_list = []
                    service_times = []
                    for i in range(self.scenario_size):
                        demands_list.append(int(np.random.poisson(5, size=1).tolist()[0]//1))
                        if demands_list[-1] > 9:
                            service_times.append(10)
                        else:
                            service_times.append(5)
                    self.add_demand(scenario, demands_list)
                    self.add_service_time(scenario, service_times)
    
    def add_capacities(self):
        for scenario, params in self.scenarios.items():
            n_realisations, _, _, _, _, cap, _, _, _, _, _ = params
            for j in range(n_realisations):
                if cap == "tight":
                    max_demand = max(self.realisations[scenario]["demands"][j])
                    self.add_capacity(scenario, max_demand)
                else:
                    max_demand = max(self.realisations[scenario]["demands"][j])
                    self.add_capacity(scenario, 3*max_demand)
    
    def add_time_windows(self):
        for scenario, params in self.scenarios.items():
            n_realisations, _, _, _, _, _, _, tw, tw_dist, _, _ = params
            for j in range(n_realisations):
                max_dist, min_dist = self.get_max_min_distance(scenario)
                if tw == "tight":
                    if tw_dist == "uniform":
                        time_windows = []
                        pseudo_median_dist = (max_dist + min_dist)/2
                        latest_latest_arrival = int(pseudo_median_dist * self.scenario_size // 2)
                        for i in range(self.scenario_size):
                            earliest_arrival = randint(0, latest_latest_arrival)
                            latest_departure = earliest_arrival + 50 + 10
                            time_windows.append((earliest_arrival, latest_departure))
                        self.add_time_window(scenario, time_windows)
                    else:
                        time_windows = []
                        r = randint(0, 10)
                        if r > 4:
                            pseudo_median_dist = (max_dist + min_dist)/2
                            latest_latest_arrival = pseudo_median_dist * self.scenario_size // 2
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, latest_latest_arrival)
                                latest_departure = earliest_arrival + 50 + 10
                                time_windows.append((earliest_arrival, latest_departure))
                            self.add_time_window(scenario, time_windows)
                        else:
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, 100)
                                latest_departure = earliest_arrival + 50 + 10
                                time_windows.append(earliest_arrival, latest_departure)
                            self.add_time_window(scenario, time_windows)
                else:
                    if tw_dist == "uniform":
                        time_windows = []
                        pseudo_median_dist = (max_dist + min_dist)/2
                        latest_latest_arrival = pseudo_median_dist * self.scenario_size // 2
                        for i in range(self.scenario_size):
                            earliest_arrival = randint(0, latest_latest_arrival)
                            latest_departure = earliest_arrival + randint(100, 400) + 10
                            time_windows.append((earliest_arrival, latest_departure))
                        self.add_time_window(scenario, time_windows)
                    else:
                        time_windows = []
                        r = randint(0, 10)
                        if r > 4:
                            pseudo_median_dist = (max_dist + min_dist)/2
                            latest_latest_arrival = pseudo_median_dist * self.scenario_size // 2
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, latest_latest_arrival)
                                latest_departure = earliest_arrival + randint(100, 400) + 10
                                time_windows.append((earliest_arrival, latest_departure))
                            self.add_time_window(scenario, time_windows)
                        else:
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, 100)
                                latest_departure = earliest_arrival + randint(100, 400) + 10
                                time_windows.append(earliest_arrival, latest_departure)
                            self.add_time_window(scenario, time_windows)

    def add_fleets(self):
        for scenario, params in self.scenarios.items():
            n_realisations, _, _, _, _, _, fleet, _, _, _, _ = params
            for j in range(n_realisations):
                max_dist, min_dist = self.get_max_min_distance(scenario)
                pseudo_median_dist = (max_dist+min_dist)/2
                cap = self.realisations[scenario]["capacities"][j]
                total_demand = sum(self.realisations[scenario]["demands"][j])
                max_departure = max(self.realisations[scenario]["time_windows"][j], key=lambda item: item[1])[0]
                r_loadxtime = total_demand/max_departure
                median_trip = pseudo_median_dist * 2 * r_loadxtime
                if fleet == "tight":
                    fleet_size = int(median_trip//cap + 1)
                    self.add_fleet(scenario, fleet_size)
                else:
                    fleet_size = int(median_trip//cap + 10)
                    self.add_fleet(scenario, fleet_size)

    def scenario_generator(self, sample_size = 20):
        location_distributions = ["double_uniform", "bivariate_normal", "double_normal", "linear_combination"] #
        lc_alpha_beta = [(0.5,0.5), (0,0.5), (0.5,0), (0.33, 0.33)] #
        depot_locations = ["central", "annular", "satelite"] #
        clusters = [0, 1, 2, 3] #
        uniform_params = [(0,100), (40, 60), (30, 40), (70, 80)]
        self.set_uniform_parameters(uniform_params[0][0], uniform_params[0][1])
        self.set_auxiliar_uniform_clusters(uniform_params[1:])
        normal_params = [
            (50, 50, 15, 5, 5, 10),
            (20, 60, 10, 15, 1, 10),
            (70, 10, 5, 15, 1, 15),
            (90, 50, 5, 5, 1, 5)
        ]
        self.set_normal_parameters(normal_params[0][0], normal_params[0][1], normal_params[0][2], normal_params[0][3], normal_params[0][4], normal_params[0][5])
        self.set_auxiliar_normal_clusters(normal_params[1:])
        demand_dist = ["constant", "uniform", "normal", "poisson"] #
        capacity_dist = ["tight", "loose"] #
        fleet_type = ["tight", "loose"] #
        tw_type = ["tight", "loose"] #
        tw_dist = ["uniform", "early"] #
        for twd in tw_dist:
            for twt in tw_type:
                for ft in fleet_type:
                    for cp in capacity_dist:
                        for dd in demand_dist:
                            for dp in depot_locations:
                                for cl in clusters:
                                    for lc in location_distributions:
                                        if lc == "linear_combination":
                                            for ab in lc_alpha_beta:
                                                name = "loc_" + lc + "-" + "AB_" + str(ab) + "-" + "cluster_" + str(cl) + "-depot_" + dp + "-demandDist_" + dd + "-capDist_" + cp + "-fleet_" + ft + "-twType_" + twt + "-twDist" + twd
                                                self.add_scenario(name, sample_size, lc, dp, cl, dd, cp, ft, twt, twd, ab[0], ab[1])

                                        else:
                                            name = "loc_" + lc + "-" + "cluster_" + str(cl) + "-depot_" + dp + "-demandDist_" + dd + "-capDist_" + cp + "-fleet_" + ft + "-twType_" + twt + "-twDist" + twd
                                            self.add_scenario(name, sample_size, lc, dp, cl, dd, cp, ft, twt, twd)

instance_object = InstanceCreator()
instance_object.scenario_generator()
instance_object.location_algo()
instance_object.add_demands()
instance_object.add_capacities()
instance_object.add_time_windows()
instance_object.add_fleets()

with open('saved_instances.pkl', 'wb') as f:
    pickle.dump(instance_object.realisations, f)

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