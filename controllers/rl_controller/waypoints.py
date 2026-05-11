"""
8 coarse waypoints — one checkpoint per road segment of the outer circuit.

Geometry extracted from city.wbt road segment translations:
  road(1) StraightRoadSegment  x=−105,  y: 4.5→−64.5   (west,  heading north)
  road(0) CurvedRoadSegment    centre (−64.5,  4.5) R=40.5  (NW curve)
  road(7) StraightRoadSegment  y=45,    x: −64.5→4.5    (north, heading east)
  road(6) CurvedRoadSegment    centre (4.5,    4.5) R=40.5  (NE curve)
  road(5) StraightRoadSegment  x=45,    y: 4.5→−64.5    (east,  heading south)
  road(4) CurvedRoadSegment    centre (4.5,   −64.5) R=40.5 (SE curve)
  road(3) StraightRoadSegment  y=−105,  x: 4.5→−64.5    (south, heading west)
  road(2) CurvedRoadSegment    centre (−64.5, −64.5) R=40.5 (SW curve)

Road width = 21.5 m → WAYPOINT_REACH_M = 12 m covers the whole road surface.

Circuit direction: CLOCKWISE.
"""

import math

def _h(t):
    """Heading on a clockwise arc at parametric angle t."""
    return math.atan2(-math.cos(t), math.sin(t))

# (x, y, heading_rad)  — one per segment, placed at each segment's midpoint
WAYPOINTS = [
    # [0] West straight midpoint  (x=−105, y mid of 4.5 and −64.5)
    (-105.0, -30.0,  math.pi / 2),

    # [1] NW curve midpoint  θ=3π/4  centre(−64.5, 4.5) R=40.5
    ( -93.1,  33.1,  _h(3 * math.pi / 4)),

    # [2] North straight midpoint  (y=45, x mid of −64.5 and 4.5)
    ( -30.0,  45.0,  0.0),

    # [3] NE curve midpoint  θ=π/4  centre(4.5, 4.5) R=40.5
    (  33.1,  33.1,  _h(math.pi / 4)),

    # [4] East straight midpoint  (x=45, y mid of 4.5 and −64.5)
    (  45.0, -30.0, -math.pi / 2),

    # [5] SE curve midpoint  θ=−π/4  centre(4.5, −64.5) R=40.5
    (  33.1, -93.1,  _h(-math.pi / 4)),

    # [6] South straight midpoint  (y=−105, x mid of 4.5 and −64.5)
    ( -30.0,-105.0,  math.pi),

    # [7] SW curve midpoint  θ=−3π/4  centre(−64.5, −64.5) R=40.5
    ( -93.1, -93.1,  _h(-3 * math.pi / 4)),
]

# Positions where barrels may be spawned (centre of right lane segments).
# Used by the procedural generation in the env reset.
# Spread across all straights of the circuit for good diversity.
BARREL_SPAWN_CANDIDATES = [
    # West straight (x=−105, going north)
    (-105.0, -55.0),
    (-105.0, -40.0),
    (-105.0, -25.0),
    (-105.0, -10.0),
    # North straight (y=45, going east)
    ( -50.0,  45.0),
    ( -35.0,  45.0),
    ( -20.0,  45.0),
    (  -5.0,  45.0),
    # East straight (x=45, going south)
    (  45.0, -10.0),
    (  45.0, -25.0),
    (  45.0, -40.0),
    (  45.0, -55.0),
    # South straight (y=−105, going west)
    ( -15.0,-105.0),
    ( -30.0,-105.0),
    ( -45.0,-105.0),
    ( -60.0,-105.0),
    # Curve midpoints
    ( -93.1,  33.1),   # NW curve
    (  33.1,  33.1),   # NE curve
    (  33.1, -93.1),   # SE curve
    ( -93.1, -93.1),   # SW curve
    # Extra west straight
    (-105.0, -70.0),
    (-105.0, -85.0),
    (-105.0, -55.0),
]

# Default (fixed) barrel positions — placed on the yellow centre line of each road.
BARREL_FIXED_POSITIONS = [
    (-105.0, -30.0, 0.6),   # west straight
    (-105.0, -55.0, 0.6),   # west straight
    ( -35.0,  45.0, 0.6),   # north straight
    (   5.0,  45.0, 0.6),   # north straight
    (  45.0, -15.0, 0.6),   # east straight
    (  45.0, -45.0, 0.6),   # east straight
    ( -20.0,-105.0, 0.6),   # south straight
    ( -55.0,-105.0, 0.6),   # south straight
]
