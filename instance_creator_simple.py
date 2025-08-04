from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
import random

# Strategy Pattern for different distributions
# This masks the distributions and the methods associated
class DistributionStrategy(ABC):
    @abstractmethod
    def generate(self) -> Any:
        pass

    @abstractmethod
    def generate_batch(self, n: int) -> List[Any]:
        pass

# Distribution section
class RandomDistribution(DistributionStrategy):
    def __init__(self, low_x = 0, low_y = 0, high_x = 100.999, high_y=100.999):
        self.low_x = low_x
        self.low_y = low_y
        self.high_x = high_x
        self.high_y = high_y
    
    def generate(self) -> Tuple[int,int]:
        from random import uniform
        x = int(uniform(self.low_x, self.high_x))
        y = int(uniform(self.low_y, self.high_y))
        return (x,y)
    
    def generate_batch(self, n: int) -> List[Tuple[int,int]]:
        sample =[]
        for _ in range(n):
            sample.append(self.generate())
        return sample

class ClusteredDistribution(DistributionStrategy):
    def __init__(self, n_seeds: int, lambda_val: float):
        self.n_seeds = n_seeds
        self.lambda_val = lambda_val
        self.pseudo_dist_vals = {}
    
    def produce_seeds(self, x_min=0, x_max=100, y_min=0, y_max=100):
        from random import uniform
        
        seeds = []
        for _ in range(self.n_seeds):
            x = int(uniform(x_min, x_max))
            y = int(uniform(y_min, y_max))
            seeds.append((x,y))
        self.seeds = seeds
    
    # 2 : calculate Uchoa function
    def pseudo_dist(self, lambda_val: float, point:Tuple[int,int]) -> float:
        from math import exp, sqrt
        val = 0
        for seed in self.seeds:
            s_x, s_y = seed
            x, y = point
            val += exp(-1/lambda_val*sqrt((s_x-x)**2+(s_y-y)**2))
        return val
    
    # 3 : get normalizing factor
    def get_normalizer(self):
        self.normalizer_min = min(self.pseudo_dist_vals.values())
        self.normalizer_max = max(self.pseudo_dist_vals.values())
    
    # 7 : Find the closest point
    def find_point(self, probability) -> Tuple[int,int]:
        return min(self.pseudo_dist_vals, key=lambda k: abs(self.pseudo_dist_vals[k] - probability))
    
    def get_empirical_dist(self):
        self.produce_seeds()
        self.pseudo_dist_vals = {(x,y): self.pseudo_dist(self.lambda_val, (x,y)) for x in range(101) for y in range(101)}
        self.get_normalizer()
        
    def generate(self) -> Tuple[int,int]:
        from random import uniform
        if self.pseudo_dist_vals == {}:
            self.get_empirical_dist()
        prob = uniform(self.normalizer_min,self.normalizer_max)
        return (self.find_point(probability=prob))
    
    def generate_batch(self, n: int) -> List[Tuple[int,int]]:
        sample =[]
        for _ in range(n):
            sample.append(self.generate())
        return sample

class UniformDistribution(DistributionStrategy):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
    
    def generate(self) -> int:
        from random import randint
        return randint(self.low, self.high)
    
    def generate_batch(self, n: int) -> List[int]:
        from random import randint
        return [randint(self.low, self.high) for _ in range(n)]

class FixedCountDistribution(DistributionStrategy):
    def __init__(self, distributions: List[DistributionStrategy], counts: List[int]):
        if len(distributions) != len(counts):
            raise ValueError("distributions and counts must have same length")
        self.distributions = distributions
        self.counts = counts
    
    def generate(self) -> Any:
        raise NotImplementedError("FixedCountDistribution requires generate_batch()")
    
    def generate_batch(self, n: int = None) -> List[Any]:
        # if n is not None:
        #     raise ValueError("n cannot be specified for FixedCountDistribution")
        
        samples = []
        for dist, count in zip(self.distributions, self.counts):
            samples.extend(dist.generate_batch(count))
        random.shuffle(samples)
        return samples

# Helper for fixed values
class FixedValueDistribution(DistributionStrategy):
    def __init__(self, value: Any):
        self.value = value

    def generate(self) -> Any:
        return self.value

    def generate_batch(self, n: int) -> List[Any]:
        return [self.value] * n

# Main class
class RandomInstanceBuilder:
    def __init__(self):
        self._properties = {}
    
    def add_property(self, name: str, distribution: DistributionStrategy) -> 'RandomInstanceBuilder':
        self._properties[name] = distribution
        return self
    
    def generate(self) -> Dict[str, Any]:
        return {name: dist.generate() for name, dist in self._properties.items()}
    
    # this is the core one
    def generate_batch(self, n: int = None, grouped: bool = True) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
        fixed_counts = any(isinstance(dist, FixedCountDistribution) 
                        for dist in self._properties.values())

        if fixed_counts:
            fixed_batches = {
                name: dist.generate_batch()
                for name, dist in self._properties.items()
                if isinstance(dist, FixedCountDistribution)
            }

            # Check consistency
            batch_sizes = [len(batch) for batch in fixed_batches.values()]
            if not all(size == batch_sizes[0] for size in batch_sizes):
                raise ValueError("All FixedCountDistribution properties must generate same number of samples")

            total_samples = batch_sizes[0]

            if grouped:
                # Return {key: [v1, ..., vn]} format
                result = {name: values[:] for name, values in fixed_batches.items()}
                for name, dist in self._properties.items():
                    if not isinstance(dist, FixedCountDistribution):
                        result[name] = [dist.generate() for _ in range(total_samples)]
                return result
            else:
                # Return list of dicts format
                data = []
                for i in range(total_samples):
                    instance = {}
                    for name, dist in self._properties.items():
                        if isinstance(dist, FixedCountDistribution):
                            instance[name] = fixed_batches[name][i]
                        else:
                            instance[name] = dist.generate()
                    data.append(instance)
                return data
        else:
            if n is None:
                raise ValueError("Must specify n when no FixedCountDistribution is used")
            
            if grouped:
                result = {name: [] for name in self._properties}
                for _ in range(n):
                    for name, dist in self._properties.items():
                        result[name].append(dist.generate())
                return result
            else:
                return [self.generate() for _ in range(n)]

class ScenarioSampler:
    def __init__(
        self,
        depot_positioning: str = "random",  # options: 'random', 'center', 'corner'
        customer_distribution: str = "random",  # options: 'random', 'clustered', 'random_clustered'
        n_customers: int = 100,
        demand_pattern: str = "unitary",  # options: 'unitary', 'u_1_10', 'u_5_10', 'u_1_100', 'u_50_100', 'mixture'
        capacity: int = 25,
        time_horizon: int = 1000,
        time_window_type: str = "tight"  # options: 'tight', 'large'
    ):
        self.depot_positioning = depot_positioning
        self.customer_distribution = customer_distribution
        self.n_customers = n_customers
        self.demand_pattern = demand_pattern
        self.capacity = capacity
        self.time_horizon = time_horizon
        self.time_window_type = time_window_type

    def build(self) -> Dict[str, Any]:
        from random import randint
        builder = RandomInstanceBuilder()

        # DEPOT
        if self.depot_positioning == "random":
            depot_dist = RandomDistribution()
        elif self.depot_positioning == "center":
            depot_dist = FixedValueDistribution((50, 50))
        elif self.depot_positioning == "corner":
            depot_dist = FixedValueDistribution((0, 0))
        else:
            raise ValueError(f"Unknown depot positioning: {self.depot_positioning}")

        builder.add_property("depot", depot_dist)

        # CUSTOMERS
        if self.customer_distribution == "random":
            customer_pos_dist = RandomDistribution()
        elif self.customer_distribution == "clustered":
            n_seeds = randint(3,8)
            customer_pos_dist = ClusteredDistribution(n_seeds=n_seeds, lambda_val=10.0)
        elif self.customer_distribution == "random_clustered":
            # mix 50% random, 50% clustered
            n1 = int(0.5 * self.n_customers)
            n2 = self.n_customers - n1
            random_dist = RandomDistribution()
            n_seeds = randint(3,8)
            clustered_dist = ClusteredDistribution(n_seeds=n_seeds, lambda_val=10.0)
            customer_pos_dist = FixedCountDistribution(
                distributions=[random_dist, clustered_dist],
                counts=[n1, n2]
            )
        else:
            raise ValueError(f"Unknown customer distribution: {self.customer_distribution}")

        if isinstance(customer_pos_dist, FixedCountDistribution):
            builder.add_property("customers", customer_pos_dist)  # Already fixed count
        else:
            builder.add_property("customers", FixedCountDistribution([customer_pos_dist], [self.n_customers]))

        # DEMANDS
        if self.demand_pattern == "unitary":
            demand_dist = FixedCountDistribution([FixedValueDistribution(1)], [self.n_customers])
        elif self.demand_pattern == "u_1_4":
            demand_dist = UniformDistribution(1, 4)
        elif self.demand_pattern == "u_3_4":
            demand_dist = UniformDistribution(3, 4)
        elif self.demand_pattern == "u_1_10":
            demand_dist = UniformDistribution(1, 10)
        elif self.demand_pattern == "u_5_10":
            demand_dist = UniformDistribution(5, 10)
        elif self.demand_pattern == "mixture":
            factor = randint(70,95)
            n1 = int(factor/100 * self.n_customers)
            n2 = self.n_customers - n1
            low = UniformDistribution(1, 4)
            high = UniformDistribution(5, 10)
            demand_dist = FixedCountDistribution(
                [low, high],
                [n1, n2]
            )
        else:
            raise ValueError(f"Unknown demand pattern: {self.demand_pattern}")

        if isinstance(demand_dist, FixedCountDistribution):
            builder.add_property("demands", demand_dist)  # Already fixed count
        else:
            builder.add_property("demands", FixedCountDistribution([demand_dist], [self.n_customers]))

        # TIME WINDOWS
        if self.time_window_type == "tight":
            window_length_dist = UniformDistribution(50, 150)
        elif self.time_window_type == "large":
            window_length_dist = UniformDistribution(200, 400)
        else:
            raise ValueError(f"Unknown time window type: {self.time_window_type}")

        earliest_arrival_dist = UniformDistribution(0, self.time_horizon)

        time_windows = []
        for _ in range(self.n_customers):
            start = earliest_arrival_dist.generate()
            length = window_length_dist.generate()
            end = min(start + length, self.time_horizon)
            time_windows.append((int(start), int(end)))

        # build instance
        instance = builder.generate_batch()

        return {
            "depot": instance["depot"][0],
            "customers": instance["customers"],
            "demands": instance["demands"],
            "capacity": self.capacity,
            "time_horizon": self.time_horizon,
            "time_windows": time_windows,
        }

class InstanceSetMaker():
    def __init__(self):
        self.DEPOT_POSITIONS = ["random", "center", "corner"]
        self.CUSTOMER_DISTRIBUTIONS = ["random", "clustered", "random_clustered"]
        self.DEMAND_PATTERNS = ["unitary", "u_1_4", "u_3_4", "u_1_10", "u_5_10", "mixture"]
        self.CAPACITIES = [10, 25, 50]
        self.TIME_HORIZONS = [1000, 2000]
        self.TIME_WINDOW_TYPES = ["tight", "large"]
        self.N_CUSTOMERS = [100]
    
    def generate_all_combinations(self, samples_per_config: int = 1) -> List[Dict[str, Any]]:
        from itertools import product
        from typing import List
        from tqdm import tqdm
        combinations = list(product(
            self.DEPOT_POSITIONS,
            self.CUSTOMER_DISTRIBUTIONS,
            self.DEMAND_PATTERNS,
            self.CAPACITIES,
            self.TIME_HORIZONS,
            self.TIME_WINDOW_TYPES,
            self.N_CUSTOMERS
        ))
        total_iterations = len(combinations) * samples_per_config
        all_instances = []
        with tqdm(total=total_iterations, desc="Generating instances") as pbar:
            for (
                depot_position,
                customer_dist,
                demand_pattern,
                capacity,
                horizon,
                time_window_type,
                n_customers
            ) in combinations:
                sampler = ScenarioSampler(
                    depot_positioning=depot_position,
                    customer_distribution=customer_dist,
                    demand_pattern=demand_pattern,
                    capacity=capacity,
                    time_horizon=horizon,
                    time_window_type=time_window_type,
                    n_customers=n_customers
                )
                for _ in range(samples_per_config):
                    instance = sampler.build()
                    instance["meta"] = {
                        "depot_positioning": depot_position,
                        "customer_distribution": customer_dist,
                        "demand_pattern": demand_pattern,
                        "capacity": capacity,
                        "time_horizon": horizon,
                        "time_window_type": time_window_type,
                        "n_customers": n_customers
                    }
                    all_instances.append(instance)
                    pbar.update(1)

        return all_instances
    
    def save_instances_as_txt(self, instances: List[Dict[str, Any]], output_dir: str):
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i, instance in enumerate(instances):
            meta = instance["meta"]
            name_parts = [
                f"depot-{meta['depot_positioning']}",
                f"dist-{meta['customer_distribution']}",
                f"demand-{meta['demand_pattern']}",
                f"cap-{meta['capacity']}",
                f"horizon-{meta['time_horizon']}",
                f"tw-{meta['time_window_type']}",
                f"n-{meta['n_customers']}",
                f"id-{i:03d}"
            ]
            instance_name = "_".join(name_parts)
            file_path = os.path.join(output_dir, f"{instance_name}.txt")

            with open(file_path, "w") as f:
                # Header
                f.write(f"{instance_name}\n\n")

                # VEHICLE info
                f.write("VEHICLE\n")
                f.write("NUMBER     CAPACITY\n")
                f.write(f"{100:>6}     {meta['capacity']}\n\n")

                # CUSTOMER info header
                f.write("CUSTOMER\n")
                f.write("CUST NO.   XCOORD.   YCOORD.   DEMAND    READY TIME   DUE DATE   SERVICE TIME\n")

                # Depot
                depot = instance["depot"]
                f.write(f"{0:>9}   {int(depot[0]):>7f}   {int(depot[1]):>7f}   {0:>6}   {0:>11}   {meta['time_horizon']:>9}   {0:>13}\n")

                # Customers
                for j, (coord, demand, tw) in enumerate(zip(instance["customers"], instance["demands"], instance["time_windows"]), start=1):
                    x, y = coord
                    ready_time, due_date = tw
                    f.write(f"{j:>9}   {int(x):>7.2f}   {int(y):>7.2f}   {int(demand):>6}   {int(ready_time):>11}   {int(due_date):>9}   {10:>13}\n")
    
set_maker = InstanceSetMaker()
instances = set_maker.generate_all_combinations(samples_per_config=15)
set_maker.save_instances_as_txt(instances, "instances/new_outputs/train")