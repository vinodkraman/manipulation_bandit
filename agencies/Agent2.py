import numpy as np
from Distributions.BernoulliDistribution import BernoulliDistribution

class Agent():
    def __init__(self, trustworthy, reputation, arm_dists, num_reports, attack):
        self.trustworthy = trustworthy
        self.reputation = reputation
        self.arm_dists = arm_dists #a array of distributions
        self.num_reports = num_reports
        self.target_arm = set(np.random.randint(len(arm_dists), size= int(len(arm_dists)/2)))
        self.attack = attack

    def generate_reports(self):
        reports = []
        for dist in self.arm_dists:
            total = 0
            for i in range(self.num_reports):
                total += dist.sample()

            reports.append(total/self.num_reports)

        return reports #returns an array of bernoulli parameters
    
    # def generate_reports_old(self):
    #     reports = []
    #     if self.trustworthy == False:
    #         for dist in self.arm_dists:
    #             total = 0
    #             for i in range(self.num_reports):
    #                 total += BernoulliDistribution(1-dist.theta).sample()

    #             reports.append(total/self.num_reports)
    #     else:
    #         for dist in self.arm_dists:
    #             total = 0
    #             for i in range(self.num_reports):
    #                 total += dist.sample()

    #             reports.append(total/self.num_reports)

    #     return reports #returns an array of bernoulli parameters
    
    # def generate_reports_random_attack(self):
    #     reports = []
    #     for index, __ in enumerate(self.arm_dists):
    #         if index == self.target_arm:
    #             reports.append(1)
    #         else:
    #             rating = np.random.rand()
    #             reports.append(rating)

    #     return reports #returns an array of bernoulli parameters

    # def generate_reports_deterministic_attack(self):
    #     reports = []
    #     for index, __ in enumerate(self.arm_dists):
    #         if index == self.target_arm:
    #             reports.append(1)
    #         else:
    #             reports.append(0)

    #     return reports #returns an array of bernoulli parameters

    
    # def generate_reports_damaging_attack(self):
    #     reports = []
    #     for dist in self.arm_dists:
    #         total = 0
    #         for i in range(self.num_reports):
    #             total += BernoulliDistribution(1-dist.theta).sample()

    #         reports.append(total/self.num_reports)
        
    #     return reports #returns an array of bernoulli parameters

    # def generate_reports_copycat_attack(self):
    #     reports = []
    #     for index, dist in enumerate(self.arm_dists):
    #         if index in self.target_arm:
    #             reports.append(1)
    #         else:
    #             total = 0
    #             for i in range(self.num_reports):
    #                 total += dist.sample()

    #             reports.append(total/self.num_reports)

    #     return reports #returns an array of bernoulli parameters


    # # def generate_reports(self):
    # #     reports = []
    # #     if self.attack == None or self.trustworthy == True:
    # #         for dist in self.arm_dists:
    # #             total = 0
    # #             for i in range(self.num_reports):
    # #                 total += dist.sample()

    # #             reports.append(total/self.num_reports)

    # #         return reports #returns an array of bernoulli parameters
    # #     elif self.attack == "random":
    # #         return self.generate_reports_random_attack()
    # #     elif self.attack == "det":
    # #         return self.generate_reports_deterministic_attack()
    # #     elif self.attack == "copy":
    # #         return self.generate_reports_copycat_attack()
    # #     elif self.attack == "dam":
    # #         return self.generate_reports_damaging_attack()