class Agent():
    def __init__(self, trustworthy, reputation, arm_dists, num_reports):
        self.trustworthy = trustworthy
        self.reputation = reputation
        self.arm_dists = arm_dists #a array of distributions
        self.num_reports = num_reports

    def generate_reports(self):
        reports = []
        for dist in self.arm_dists:
            total = 0
            for i in range(self.num_reports):
                total += dist.sample()

            reports.append(total/self.num_reports)

        return reports #returns an array of bernoulli parameters


#truthful to one arm, malicious to another?