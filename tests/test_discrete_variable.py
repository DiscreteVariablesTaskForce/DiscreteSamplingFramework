import pytest
from discretesampling.base.types import DiscreteVariable


class ExampleParticleClass(DiscreteVariable):
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x

    def getProposalType(self):
        return super().getProposalType()

    def getTargetType(self):
        return super().getTargetType()


@pytest.mark.parametrize(
    "x",
    [(ExampleParticleClass(1)),
     (ExampleParticleClass(10))]
)
def test_encode_decode_generic_type(x):
    encoded_x = ExampleParticleClass.encode(x)
    decoded_x = ExampleParticleClass.decode(encoded_x, x)
    assert x == decoded_x
