"""Legacy stubs kept only so old config versions fail loudly instead of at import.

RiTAL (model ``version: 3``) uses :class:`rital.model.RiTALModel`. The RNA-FM
variants below were never part of the released model — they are stubs that the
trainer's version dispatch still references.
"""
from rital.model import RiTALModel


class RiTALModelRNAFM(RiTALModel):
    """Stub for the old ``version: 4`` (RNA-FM) branch — not included."""

    def rna_fm_unfrozen_params(self):
        return []


class RiTALModelRNAFMPlus(RiTALModel):
    """Stub for the old ``version: 10`` (RNA-FM+) branch — not included."""

    def rna_fm_unfrozen_params(self):
        return []


# OptiMolLigandEncoder lives in rital.optimol to avoid a circular import
# (model.py needs it at import time, and this module imports model.py).
