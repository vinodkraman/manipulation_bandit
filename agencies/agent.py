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
        self.second_best = 0
        self.rewards = []

    def compute_logit(self, d):
        return np.log(d/(1-d))

    def add_noise(self, data, sigma= 0):
        noise = np.random.normal(0, sigma, len(self.arm_dists))
        logits = [self.compute_logit(param) for param in data]
        new_hidden_logits = noise + logits
        final = [1/(1+np.exp(-val)) for val in new_hidden_logits]
        return final


    def add_biased_noise(self, data, sigma= 0):
        mult = np.array([-1 if elem == 0 else 1 for elem in self.rewards])
        noise = np.array([np.random.normal(0, sigma) for elem in mult])
        # noise = np.random.normal(0, sigma, len(self.arm_dists))
        # mult = np.array([-1 if elem == 0 else 1 for elem in self.rewards])
        # noise = np.multiply(noise, mult) 
        logits = np.array([self.compute_logit(param) for param in data])
        new_hidden_logits = noise + logits
        final = [1/(1+np.exp(-val)) for val in new_hidden_logits]
        return final

    def add_binomial_noise(self, data):
        test = [np.random.binomial(self.num_reports, param)/self.num_reports for param in data]
        return test

    def add_binary_noise(self, data, noise_param= 0):
        noise = []
        if noise_param == 0:
            noise = np.array([0 for reward in data])
        elif noise_param == 1:
            noise = np.array([1 for reward in data])
        else:
            noise = np.array([BernoulliDistribution(noise_param).sample() for reward in data])
        noise_reward = data + noise
        noise_reward = np.remainder(noise_reward, 2)
        return noise_reward

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

    def generate_reports_sneak_attack(self):
        data = [dist.mean() for dist in self.arm_dists]
        # arg_sorted = np.argsort(data)
        reports = self.add_biased_noise(data)
        # reports = np.array(reports)
        # reports = 1 - reportsccc
        reports[self.best_arm] = 0
        # reports[self.worst_arm] = 1
        # reports[arg_sorted[-2]]
        
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
                
                return reports
        
        return self.generate_reports_sneak_attack()

    def generate_reports_sleeper_attack(self, t, prev_agents, prev_agent_reports, attack="sneak"):
        reports = []
        if np.random.rand() < self.attack_freq:
            if attack == "copy":
                return self.generate_reports_copy_cat_attack(prev_agents, prev_agent_reports)
            elif attack == "damage":
                return self.generate_reports_max_damage()
            elif attack == "sneak":
                return self.generate_reports_sneak_attack()
        else:
            data = [dist.mean() for dist in self.arm_dists]
            reports = self.add_biased_noise(data)
            return reports
            # for dist in self.arm_dists:
            #     reports.append(dist.mean())
            # return reports

    def generate_reports_prolonged_attack(self, t, prev_agents, prev_agent_reports, attack="damage"):
        if (t > 500):
            return self.generate_reports_sneak_attack()
        else:
            data = [dist.mean() for dist in self.arm_dists]
            # reports = self.add_binary_noise(self.rewards)
            # reports = self.add_noise(data)
            reports = self.add_biased_noise(data)
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
            reports.append(1-dist.mean())

        return reports #returns an array of bernoulli parameters

    def generate_reports_random_attack(self):
        reports = []
        for index, __ in enumerate(self.arm_dists):
            if index == self.best_arm:
                reports.append(0)
            else:
                reports.append(np.random.rand())

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
        data = [dist.mean() for dist in self.arm_dists]
        if self.trustworthy == True:
            # reports = self.add_noise(data)
            # reports = self.add_binary_noise(self.rewards)
            reports = self.add_biased_noise(data)
            
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