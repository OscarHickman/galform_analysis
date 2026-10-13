"""Parameter sets for GALFORM's dust model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DustModel:
    """Parameters of GALFORM's diffuse-dust model.

    Attributes:
        name: Label for the model.
        table: GALFORM dust table: a file name in ``<GALFORM_DIR>/Data/dust``,
            or a path.
        fcloud: Fraction of the cold gas (and its dust) in molecular clouds;
            the remaining ``1 - fcloud`` is diffuse.
        rfacburst: Radius of a burst's dust disk over the bulge half-mass
            radius; 0 ignores dust in bursts.
    """

    name: str
    table: str
    fcloud: float
    rfacburst: float

    def __post_init__(self):
        if not 0.0 <= self.fcloud <= 1.0:
            raise ValueError(f"fcloud must be in [0, 1], got {self.fcloud}")
        if self.rfacburst < 0.0:
            raise ValueError(f"rfacburst must be >= 0, got {self.rfacburst}")


#: Lacey et al. (2016) (GALFORM ``dustpars_Lacey16.csh``; the lc16 model).
LACEY16 = DustModel(
    name="lacey16", table="dust_MW_hz1.0.dat", fcloud=0.5, rfacburst=1.0
)
#: Baugh et al. (2005).
BAUGH05 = DustModel(
    name="baugh05", table="dust_MW_hz1.0.dat", fcloud=0.25, rfacburst=1.0
)

DUST_MODELS = {model.name: model for model in (LACEY16, BAUGH05)}
