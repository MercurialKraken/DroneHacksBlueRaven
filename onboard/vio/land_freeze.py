"""Hold the vision pose still while PX4 lands, once the rangefinder says the floor is close."""
from __future__ import annotations

from typing import Any

NAV_DESCEND, NAV_AUTO_LAND = 12, 18   # px4_msgs VehicleStatus.NAVIGATION_STATE_*
LANDING_NAV = (NAV_DESCEND, NAV_AUTO_LAND)


class LandFreeze:
    """Latches the last published sample; only disarm or leaving the landing modes releases it."""

    def __init__(self, freeze_m: float, fresh_s: float = 0.3) -> None:
        if not (freeze_m > 0 and fresh_s > 0):raise ValueError('land_freeze_m and fresh_s must be positive')
        self.freeze_m,self.fresh_s=freeze_m,fresh_s
        self.frozen: dict[str, Any] | None = None

    def update(self, nav: int | None, armed: bool, range_m: float | None, range_age_s: float | None,
               sample: dict[str, Any] | None) -> dict[str, Any] | None:
        if not (armed and nav in LANDING_NAV):
            self.frozen=None
        elif (self.frozen is None and sample is not None and range_m is not None and range_age_s is not None
              and 0<=range_age_s<self.fresh_s and range_m<=self.freeze_m):
            self.frozen=sample
        return self.frozen
