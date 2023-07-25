import pytest
from discretesampling.domain.additive_structure.additive_structure import AdditiveStructure


@pytest.mark.parametrize(
    "ad",
    [(AdditiveStructure([[1, 2], [3, 4, 5]])),
     (AdditiveStructure([[1, 2, 3], [4, 5]]))]
)
def test_encode_decode_additivestructure(ad):
    encoded_ad = ad.encode(ad)
    decoded_ad = ad.decode(encoded_ad, ad)
    assert ad == decoded_ad
