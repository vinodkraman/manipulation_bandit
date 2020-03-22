from distributions.bernoullidistribution import BernoulliDistribution
import numpy as np
import copy
class Agent():
    def __init__(self, trustworthy, arm_dists, num_reports, best_arm, target_arms, attack_freq, agent_id):
        self.trustworthy = trustworthy
        self.arm_dists = arm_dists #a array of distributions
        self.num_reports = num_reports
        self.best_arm = best_arm
        self.worst_arm = 0
        self.target_arms = target_arms
        self.attack_freq = attack_freq
        self.id = agent_id

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

    def generate_reports_sneak_attack(self):
        reports = []
        for index, dist in enumerate(self.arm_dists):
            if index == self.best_arm:
                reports.append(0.5)
            # elif index == self.worst_arm:
            #     reports.append(1)
            else:
                # reports.append(np.mean(dist.sample_array(self.num_reports)))
                reports.append(dist.mean())


        return reports #returns an array of bernoulli parameters


    def generate_reports(self):
        if self.trustworthy == True:
            reports = []
            for dist in self.arm_dists:
                reports.append(np.mean(dist.sample_array(self.num_reports)))

            return reports #returns an array of bernoulli parameters
        else:
            # print("ello")
            return self.generate_reports_sneak_attack()

    def generate_reports_copy_cat_attack(self, prev_agents, prev_agent_reports):
        reports = []
        for agent_index, agent in enumerate(prev_agents):
            if agent.trustworthy == True:
                reports = copy.deepcopy(prev_agent_reports[agent])
                reports[self.best_arm] = 0.5
                # for target_arm_index in self.target_arms:
                #     reports[target_arm_index] = 1
                
                return reports
        
        return self.generate_reports_sneak_attack()

    def generate_reports_sleeper_attack(self, t, prev_agents, prev_agent_reports, attack="sneak"):
        reports = []
        if np.random.rand() >= self.attack_freq:
            if attack == "copy":
                return self.generate_reports_copy_cat_attack(prev_agents, prev_agent_reports)
            elif attack == "damage":
                return self.generate_reports_max_damage()
            elif attack == "sneak":
                return self.generate_reports_sneak_attack()
        else:
            for dist in self.arm_dists:
                reports.append(dist.mean())
                # reports.append(np.mean(dist.sample_array(self.num_reports)))
            return reports

    def generate_reports_prolonged_attack(self, t, prev_agents, prev_agent_reports, attack="damage"):
        reports = []
        if (t > 100 and t < 200) or (t > 400 and t < 500) or (t > 700 and t < 800):
            # print(t)
            return self.generate_reports_sneak_attack()
        else:
            for dist in self.arm_dists:
                reports.append(np.mean(dist.sample_array(self.num_reports)))
            return reports

    def generate_reports_average_attack(self, prev_agents, prev_agent_reports):
        reports = np.zeros(len(self.arm_dists))
        count = 0

        if len(prev_agents) == 0:
            return self.generate_reports_sneak_attack()
        else:
            for agent_index, _ in enumerate(prev_agents):
                reports = np.add(reports, prev_agent_reports[agent_index])
                count += 1

            reports /= count
            reports[self.best_arm] = 0
            for target_arm_index in self.target_arms:
                reports[target_arm_index] = 1
            return reports.tolist()

    def generate_reports_max_damage(self):
        reports = []
        for dist in self.arm_dists:
            reports.append(np.mean(BernoulliDistribution(1-dist.theta).sample_array(self.num_reports)))
        
        return reports #returns an array of bernoulli parameters

    def generate_reports_random_attack(self):
        reports = []
        for index, __ in enumerate(self.arm_dists):
            if index in self.target_arms:
                reports.append(1)
            else:
                rating = np.random.rand()
                reports.append(rating)

        return reports #returns an array of bernoulli parameters

    def generate_reports_deterministic_attack(self):
        reports = []
        for index, __ in enumerate(self.arm_dists):
            if index in self.target_arms:
                reports.append(1)
            else:
                reports.append(0)

        return reports #returns an array of bernoulli parameters

    def generate_reports_v2(self, t, attack, prev_agents= [], prev_agent_reports= []):
        # [print(dist.mean()) for dist in self.arm_dists]
        if self.trustworthy == True:
            reports = []
            for dist in self.arm_dists:
                reports.append(np.mean(dist.sample_array(self.num_reports)))
            
            return reports #returns an array of bernoulli parameters
        else:
            if attack == "copy":
                return self.generate_reports_copy_cat_attack(prev_agents, prev_agent_reports)
            elif attack == "avg":
                return self.generate_reports_average_attack(prev_agents, prev_agent_reports)
            elif attack == "sneak":
                return self.generate_reports_sneak_attack()
            elif attack == "damage":
                return self.generate_reports_max_damage()
            elif attack == "deterministic":
                return self.generate_reports_deterministic_attack()
            elif attack == "sleeper":
                return self.generate_reports_sleeper_attack(t, prev_agents, prev_agent_reports)
            elif attack == "random":
                return self.generate_reports_random_attack()
            elif attack == "prolonged":
                return self.generate_reports_prolonged_attack(t, prev_agents, prev_agent_reports)
            else:
                exit()