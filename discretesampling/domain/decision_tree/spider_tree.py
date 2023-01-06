from .tree import Tree
from .spider_distribution import SpiderTreeProposal

class SpiderTree(Tree):
    @classmethod
    def getProposalType(self):
        return SpiderTreeProposal