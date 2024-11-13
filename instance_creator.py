import numpy as np
from random import randint
from math import sqrt, ceil
from sys import float_info
from statistics import mean
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
        n_sc = len(self.scenarios)
        cnt = 0
        for scenario, params in self.scenarios.items():
            print("Adding demands "+str(cnt)+" of "+str(n_sc))
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
            cnt += 1
    
    def add_capacities(self, tight_factor=5, loose_factor=20):
        n_sc = len(self.scenarios)
        cnt = 0
        for scenario, params in self.scenarios.items():
            print("Adding capacities "+str(cnt)+" of "+str(n_sc))
            n_realisations, _, _, _, _, cap, _, _, _, _, _ = params
            for j in range(n_realisations):
                if cap == "tight":
                    sum_demand = sum(self.realisations[scenario]["demands"][j])
                    self.add_capacity(scenario, ceil(tight_factor/self.scenario_size * sum_demand)) # from Uchoa et al. New benchmark instances for the Capacitated Vehicle Routing Problem
                else:
                    sum_demand = sum(self.realisations[scenario]["demands"][j])
                    self.add_capacity(scenario, ceil(loose_factor/self.scenario_size * sum_demand)) # from Uchoa et al. New benchmark instances for the Capacitated Vehicle Routing Problem
            cnt += 1
    
    def add_time_windows(self):
        n_sc = len(self.scenarios)
        cnt = 0
        for scenario, params in self.scenarios.items():
            print("Adding TW "+str(cnt)+" of "+str(n_sc))
            n_realisations, _, _, _, _, _, _, tw, tw_dist, _, _ = params
            for j in range(n_realisations):
                max_dist, min_dist = self.get_max_min_distance(scenario)
                if tw == "tight":
                    if tw_dist == "uniform":
                        time_windows = []
                        pseudo_median_dist = (max_dist + min_dist)/2
                        latest_latest_arrival = int(pseudo_median_dist * self.scenario_size // 4)
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
                            latest_latest_arrival = int(pseudo_median_dist * self.scenario_size // 4)
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, latest_latest_arrival)
                                latest_departure = earliest_arrival + 50 + 10
                                time_windows.append((earliest_arrival, latest_departure))
                            self.add_time_window(scenario, time_windows)
                        else:
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, 100)
                                latest_departure = earliest_arrival + 50 + 10
                                time_windows.append((earliest_arrival, latest_departure))
                            self.add_time_window(scenario, time_windows)
                else:
                    if tw_dist == "uniform":
                        time_windows = []
                        pseudo_median_dist = (max_dist + min_dist)/2
                        latest_latest_arrival = int(pseudo_median_dist * self.scenario_size // 4)
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
                            latest_latest_arrival = int(pseudo_median_dist * self.scenario_size // 4)
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, latest_latest_arrival)
                                latest_departure = earliest_arrival + randint(100, 400) + 10
                                time_windows.append((earliest_arrival, latest_departure))
                            self.add_time_window(scenario, time_windows)
                        else:
                            for i in range(self.scenario_size):
                                earliest_arrival = randint(0, 100)
                                latest_departure = earliest_arrival + randint(100, 400) + 10
                                time_windows.append((earliest_arrival, latest_departure))
                            self.add_time_window(scenario, time_windows)
            cnt += 1

    def add_fleets(self):
        n_sc = len(self.scenarios)
        cnt = 0
        for scenario, params in self.scenarios.items():
            print("Adding fleet "+str(cnt)+" of "+str(n_sc))
            n_realisations, _, _, _, _, _, fleet, _, _, _, _ = params
            for j in range(n_realisations):
                cap = self.realisations[scenario]["capacities"][j]
                demands = self.realisations[scenario]["demands"][j]
                total_demand = sum(demands)
                average_demand = mean(demands)
                delta = 60
                width = self.ub - self.lb
                if fleet == "tight":
                    fleet_size = int(ceil((total_demand*average_demand)/(cap**2)) +  ceil((self.scenario_size*delta**2)/(cap*width**2)))
                    self.add_fleet(scenario, fleet_size)
                else:
                    fleet_size = int(ceil(1.2*(total_demand*average_demand)/(cap**2)) +  ceil(1.2*(self.scenario_size*delta**2)/(cap*width**2)))
                    self.add_fleet(scenario, fleet_size)
            cnt += 1

    def scenario_generator(self, sample_size = 4, location_distributions = [], depot_locations = [], clusters = [], uniform_params = [], normal_params = [], demand_dist = [], capacity_dist = [], fleet_type = [], tw_type = [], tw_dist = []):
        
        if location_distributions == []:
            location_distributions = ["double_uniform", "bivariate_normal", "double_normal", "linear_combination"] #
        lc_alpha_beta = [(0.5,0.5), (0,0.5), (0.5,0), (0.33, 0.33)] #
        if depot_locations == []:
            depot_locations = ["central", "annular", "satelite"] #
        if clusters == []:
            clusters = [0, 1, 2, 3] #
        if uniform_params == []:
            uniform_params = [(0,100), (40, 60), (30, 40), (70, 80)]
        self.set_uniform_parameters(uniform_params[0][0], uniform_params[0][1])
        self.set_auxiliar_uniform_clusters(uniform_params[1:])
        if normal_params == []:
            normal_params = [
                (50, 50, 15, 5, 5, 10),
                (20, 60, 10, 15, 1, 10),
                (70, 10, 5, 15, 1, 15),
                (90, 50, 5, 5, 1, 5)
            ]
        self.set_normal_parameters(normal_params[0][0], normal_params[0][1], normal_params[0][2], normal_params[0][3], normal_params[0][4], normal_params[0][5])
        self.set_auxiliar_normal_clusters(normal_params[1:])
        if demand_dist == []:
            demand_dist = ["constant", "uniform", "normal", "poisson"] #
        if capacity_dist == []:
            capacity_dist = ["tight", "loose"] #
        if fleet_type == []:
            fleet_type = ["tight", "loose"] #
        if tw_type == []:
            tw_type = ["tight", "loose"] #
        if tw_dist == []:
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

instance_object = InstanceCreator(scenario_size=100)
instance_object.scenario_generator(
    sample_size=1, 
    location_distributions=["linear_combination"], 
    depot_locations=["annular"], 
    clusters=[0], 
    demand_dist=["poisson"], 
    capacity_dist=["tight"], 
    fleet_type=["loose"], 
    tw_dist=["uniform"], 
    tw_type=["loose"]
    )
instance_object.location_algo()
instance_object.add_demands()
tight_factor = 20
loose_factor = 200
instance_object.add_capacities(tight_factor, loose_factor)
instance_object.add_time_windows()
instance_object.add_fleets()

# with open('saved_instances.pkl', 'wb') as f:
#     pickle.dump(instance_object.realisations, f)

# Create instances
output_dir = 'outputs9'

counter = -1
for i in instance_object.realisations:
    for j in range(len(instance_object.realisations[i]["locations"])):
        counter += 1
        name = 'JU6H{:06d}'.format(counter)
        max_dep = max(instance_object.realisations[i]["time_windows"][j], key=lambda item: item[1])[0]
        vehicles = instance_object.realisations[i]["fleets"][j]
        capacity = instance_object.realisations[i]["capacities"][j]
        customers = instance_object.realisations[i]["locations"][j]
        text = '//' + i + '\n' + name +'\n\nVEHICLE\nNUMBER     CAPACITY\n'
        v_format = "    "
        v_format = v_format[0:4-len(str(vehicles))] + str(vehicles)
        c_format = "           "
        c_format = c_format[0:11-len(str(capacity))] + str(capacity)
        text += v_format + "  " + c_format + "\n\nCUSTOMER\nCUST NO.   XCOORD.   YCOORD.   DEMAND    READY TIME   DUE DATE   SERVICE TIME\n\n"
        cust_id_format = "   "
        cust_x_format = "      "
        cust_y_format = "         "
        cust_d_format = "         "
        cust_arr_format = "         "
        cust_dep_format = "         "
        cust_st_format = "         "
        cust_id_format = cust_id_format[0:3-len(str(0))] + str(0)
        cust_x_format = cust_x_format[0:6-len(str(instance_object.realisations[i]["depots"][0][0]))] + str(instance_object.realisations[i]["depots"][0][0])
        cust_y_format = cust_y_format[0:9-len(str(instance_object.realisations[i]["depots"][0][1]))] + str(instance_object.realisations[i]["depots"][0][1])
        cust_d_format = cust_d_format[0:9-len(str(0))] + str(0)
        cust_arr_format = cust_arr_format[0:9-len(str(0))] + str(0)
        cust_dep_format = cust_dep_format[0:9-len(str(max_dep + 300))] + str(max_dep+300)
        cust_st_format = cust_st_format[0:9-len(str(0))] + str(0)
        text += "  "+cust_id_format+"  "+cust_x_format+"  "+cust_y_format+"  "+cust_d_format+"  "+cust_arr_format+"  "+cust_dep_format+"  "+cust_st_format+"\n"
        for cust_idx in range(len(customers)):
            cust_id_format = "   "
            customer_latlon = customers[cust_idx]
            customer_demand = instance_object.realisations[i]["demands"][j][cust_idx]
            arrdep = instance_object.realisations[i]["time_windows"][j][cust_idx]
            stimes = instance_object.realisations[i]["service_times"][j][cust_idx]
            cust_id_format = cust_id_format[0:3-len(str(cust_idx+1))] + str(cust_idx+1)
            cust_x_format = "      "
            cust_x_format = cust_x_format[0:6-len(str(customer_latlon[0]))] + str(customer_latlon[0])
            cust_y_format = "         "
            cust_y_format = cust_y_format[0:9-len(str(customer_latlon[1]))] + str(customer_latlon[1])
            cust_d_format = "         "
            cust_d_format = cust_d_format[0:9-len(str(customer_demand))] + str(customer_demand)
            cust_arr_format = "         "
            cust_arr_format = cust_arr_format[0:9-len(str(arrdep[0]))] + str(arrdep[0])
            cust_dep_format = "         "
            cust_dep_format = cust_dep_format[0:9-len(str(arrdep[1]))] + str(arrdep[1])
            cust_st_format = "         "
            cust_st_format = cust_st_format[0:9-len(str(stimes))] + str(stimes)
            text += "  "+cust_id_format+"  "+cust_x_format+"  "+cust_y_format+"  "+cust_d_format+"  "+cust_arr_format+"  "+cust_dep_format+"  "+cust_st_format+"\n"
        with open("{}/{}.txt".format(output_dir,name), "w") as text_file:
            text_file.write(text)
            text_file.close()

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