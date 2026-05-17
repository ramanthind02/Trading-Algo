from __future__ import annotations

from typing import NewType

from quantfoundry_core.zone_manager.models import ZoneType

TestId = NewType("TestId", str)


def allowed_robustness_tests(zone_type: ZoneType) -> frozenset[TestId]:
    """Catalog of robustness tests permitted for a **strategy-tier** zone type.

    Portfolio / project test evaluation is not selected via ZoneType; see
    ``docs/SaaS/zone_manager.md`` §5 and §7.5.

    MVP: placeholder ids for Worker/API wiring.
    """
    train = frozenset({TestId("permutation_placeholder"), TestId("complexity_aware_placeholder")})
    validation_oos = frozenset(
        {TestId("return_shuffle_placeholder"), TestId("monte_carlo_placeholder")}
    )
    match zone_type:
        case ZoneType.TRAIN:
            return train
        case ZoneType.VALIDATION:
            return validation_oos
