from BetaDistribution import BetaDistribution

# def mean(self):
#         return (self.alpha)/ (self.alpha + self.beta)
    
#     def update(self, delta_alpha, delta_beta):
#         self.alpha += delta_alpha
#         self.beta += delta_beta

#     def sample(self):
#         return np.asscalar(beta.rvs(self.alpha, self.beta, size=1))

#     def get_params(self):
#         return self.alpha, self.beta
#test
beta1 = BetaDistribution(1,3)
assert (beta1.mean() == 0.25)

beta1 = BetaDistribution(3,2)
assert (beta1.mean() == 3/5)

a, b = beta1.get_params()
assert(a == 3)
assert(b == 2)

beta1.update(1, 9)
a, b = beta1.get_params()
assert(a == 4)
assert(b == 11)
assert (beta1.mean() == 4/15)

num_samples = 100000
total = 0
for i in range(num_samples):
    total += beta1.sample()

print(total/num_samples)
assert(4/15 - 0.03 <= total/num_samples <= 4/15 + 0.03)