from discretesampling.domain import coin
from discretesampling.base.algorithms import DiscreteVariableMCMC, DiscreteVariableSMC

target = coin.CoinStackTarget(10, 5)  # Observed 10 heads from some noisy counting

# num_coins~U(1,5) coins with probability 0.9, else num_coins~U(6,15)
initialProposal = coin.CoinStackInitialProposal(5, 15, 0.9)

coinMCMC = DiscreteVariableMCMC(coin.CoinStack, target, initialProposal)
try:
    samples = coinMCMC.sample(100)

except ZeroDivisionError:
    print("MCMC sampling failed due to division by zero")


coinSMC = DiscreteVariableSMC(coin.CoinStack, target, initialProposal)
try:
    SMCSamples = coinSMC.sample(2, 50)

except ZeroDivisionError:
    print("SMC sampling failed due to division by zero")

coins_MCMC = [len(x.list_of_coin_tosses) for x in samples[50:99]]
coins_SMC = [len(x.list_of_coin_tosses) for x in SMCSamples]

heads_MCMC = [sum(x.list_of_coin_tosses) for x in samples[50:99]]
heads_SMC = [sum(x.list_of_coin_tosses) for x in SMCSamples]

print("Samples of estimated number of coins, MCMC: ", coins_MCMC, "\n")
print("Samples of estimated number of coins, SMC: ", coins_SMC, "\n")

print("Samples of estimated number of heads, MCMC: ", heads_MCMC, "\n")
print("Samples of estimated number of heads, SMC: ", heads_SMC, "\n")
