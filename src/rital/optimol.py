"""Stub: OptiMol pretrained ligand encoder. Only needed when use_pretrained_ligand=True.

RiTAL's released configs all set ``use_pretrained_ligand: false``.
"""


class OptiMolLigandEncoder:
    def __init__(self, out_dim=256, freeze=True, optimol_dir=None, map_file=None):
        raise RuntimeError(
            "OptiMol encoder not available. Set use_pretrained_ligand: false in config."
        )
