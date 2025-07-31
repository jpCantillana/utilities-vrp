from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np
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
    def __init__(self, n_seeds: int, n_points: int, lambda_val: float):
        self.n_seeds = n_seeds
        self.n_points = n_points
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
        self.get_normalizer(self.pseudo_dist_vals)
        
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

class FixedCountDistribution(DistributionStrategy):
    def __init__(self, distributions: List[DistributionStrategy], counts: List[int]):
        if len(distributions) != len(counts):
            raise ValueError("distributions and counts must have same length")
        self.distributions = distributions
        self.counts = counts
    
    def generate(self) -> Any:
        raise NotImplementedError("FixedCountDistribution requires generate_batch()")
    
    def generate_batch(self, n: int = None) -> List[Any]:
        if n is not None:
            raise ValueError("n cannot be specified for FixedCountDistribution")
        
        samples = []
        for dist, count in zip(self.distributions, self.counts):
            samples.extend(dist.generate_batch(count))
        random.shuffle(samples)
        return samples

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
    def generate_batch(self, n: int = None) -> List[Dict[str, Any]]:
        fixed_counts = any(isinstance(dist, FixedCountDistribution) 
                          for dist in self._properties.values())
        
        if fixed_counts:
            if n is not None:
                raise ValueError("Cannot specify n when using FixedCountDistribution")
            
            # Generate all fixed-count batches only once
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
            
            # Generate data
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
            return [self.generate() for _ in range(n)]
        
# Define distributions
age_dist = FixedCountDistribution(
    distributions=[
        UniformDistribution(30, 40),
        UniformDistribution(60, 70)
    ],
    counts=[2, 2] 
)

income_dist = NormalDistribution(50_000, 10_000)

# Build and generate
builder = RandomInstanceBuilder() \
    .add_property("age", age_dist) \
    .add_property("income", income_dist)

people = builder.generate_batch()  # Automatically generates 60 samples

# Verify
print(people)  # 4
# ages = [p["age"] for p in people]
# print(sum(18 <= age <= 25 for age in ages))  # Should be ~30 from Uniform