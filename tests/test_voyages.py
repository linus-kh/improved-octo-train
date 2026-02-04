import math

from ais_portwatch_delay_risk.config import BBox
from ais_portwatch_delay_risk.voyages import haversine_m, in_bbox


def test_in_bbox():
    bbox = BBox(lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    assert in_bbox(0.5, 0.5, bbox)
    assert not in_bbox(-0.1, 0.5, bbox)


def test_haversine_m_equator_degree():
    dist = haversine_m(0.0, 0.0, 0.0, 1.0)
    assert math.isclose(dist, 111_319.9, rel_tol=0.02)
