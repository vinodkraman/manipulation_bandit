from distributions.bernoullidistribution import BernoulliDistribution
import numpy as np
class Agent():
    def __init__(self, trustworthy, reputation, arm_dists, num_reports, best_arm, target_arms):
        self.trustworthy = trustworthy
        self.reputation = reputation
        self.arm_dists = arm_dists #a array of distributions
        self.num_reports = num_reports
        self.best_arm = best_arm
        self.target_arms = target_arms

    # def generate_reports(self):
    #     reports = []
    #     for dist in self.arm_dists:
    #         total = 0
    #         for i in range(self.num_reports):
    #             total += dist.sample()

    #         reports.append(total/self.num_reports)

    #     return reports #returns an array of bernoulli parameters
    # def generate_reports(self):
    #     reports = []
    #     for dist in self.arm_dists:
    #         total = 0
    #         for i in range(self.num_reports):
    #             if not self.trustworthy:
    #                 total += BernoulliDistribution(1-dist.theta).sample()
    #             else:
    #                 total += dist.sample()

    #         reports.append(total/self.num_reports)

    #     return reports #returns an array of bernoulli parameters

    def generate_reports_2(self):
        reports = []
        for dist in self.arm_dists:
            total = 0
            for _ in range(self.num_reports):
                if not self.trustworthy:
                    total += BernoulliDistribution(1-dist.theta).sample()
                else:
                    total += dist.sample()

            reports.append(total/self.num_reports)

        return reports #returns an array of bernoulli parameters

    # def generate_reports(self):
    #     reports = []
    #     for index, dist in enumerate(self.arm_dists):
    #         if index == self.best_arm and self.trustworthy == False:
    #             reports.append(0)
    #             # print("Hello")
    #         elif index == self.target_arm and self.trustworthy == False:
    #             reports.append(1)
    #             # print("Goodbye")
    #         else:
    #             total = 0
    #             for i in range(self.num_reports):
    #                 total += dist.sample()

    #             reports.append(total/self.num_reports)

    #     return reports #returns an array of bernoulli parameters

    def generate_reports_copycat_attack(self):
        reports = []
        for index, dist in enumerate(self.arm_dists):
            if index == self.best_arm:
                reports.append(0)
            elif index in self.target_arms:
                reports.append(1)
            else:
                reports.append(np.mean(dist.sample_array(self.num_reports)))

        return reports #returns an array of bernoulli parameters


    def generate_reports(self):
        if self.trustworthy == True:
            reports = []
            for dist in self.arm_dists:
                reports.append(np.mean(dist.sample_array(self.num_reports)))

            return reports #returns an array of bernoulli parameters
        else:
            # print("ello")
            return self.generate_reports_copycat_attack()