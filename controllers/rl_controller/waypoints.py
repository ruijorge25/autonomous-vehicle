"""
Waypoints for the right-hand lane of the City world outer circuit.

Each waypoint is (x, y, heading_rad) where heading is the road direction
at that point (the direction the vehicle should be travelling).

The circuit is approximately rectangular, driven anti-clockwise:
  Start near (-83, -102) heading ~north  →  top-left corner  →
  top-right corner  →  bottom-right corner  →  back south.

Lane offset: the road is 21.5 m wide, 4 lanes, so each lane is ~5.375 m.
Right-hand lane centre is offset 2.69 m to the right of the road centre line.

Coordinate system: Webots X = east, Y = north.

The waypoints below were derived from the StraightRoadSegment and
CurvedRoadSegment translations in city.wbt.  They are spaced ~10 m apart
on the straight sections and every ~15 deg on the curves.
"""

import math

# (x, y, heading_rad)
# heading = angle of travel in radians, measured from +X axis (east),
# counter-clockwise positive (Webots standard).
WAYPOINTS = [
    # ── South straight (road(1)): x≈-105, going north (heading=+π/2)
    (-108.69, -102.0,  math.pi / 2),
    (-108.69,  -92.0,  math.pi / 2),
    (-108.69,  -82.0,  math.pi / 2),
    (-108.69,  -72.0,  math.pi / 2),
    (-108.69,  -62.0,  math.pi / 2),
    (-108.69,  -52.0,  math.pi / 2),
    (-108.69,  -42.0,  math.pi / 2),
    (-108.69,  -32.0,  math.pi / 2),
    (-108.69,  -22.0,  math.pi / 2),
    (-108.69,  -12.0,  math.pi / 2),
    (-108.69,   -2.0,  math.pi / 2),

    # ── NW curve (road(0)): centre (-64.5, 4.5), turning from north to east
    (-100.0,    7.19,  math.pi / 2 - math.radians(15)),
    ( -93.0,   10.5,   math.pi / 2 - math.radians(30)),
    ( -85.0,   13.3,   math.pi / 2 - math.radians(50)),
    ( -76.5,   14.8,   math.pi / 2 - math.radians(70)),
    ( -67.8,   14.8,   math.pi / 2 - math.radians(85)),
    ( -61.5,   13.3,   0.0 + math.radians(15)),
    ( -55.5,   10.0,   0.0 + math.radians(5)),

    # ── North straight (road(7)): y≈45, going east (heading=0)
    ( -45.0,    7.19,  0.0),
    ( -35.0,    7.19,  0.0),
    ( -25.0,    7.19,  0.0),

    # ── NE inner curve (road(6)): centre (4.5, 4.5), turning east to south
    ( -15.0,    7.19,  0.0 - math.radians(10)),
    (  -5.0,    5.5,   0.0 - math.radians(25)),
    (   2.5,    1.5,   0.0 - math.radians(50)),
    (   5.5,   -5.0,  -math.pi / 2 + math.radians(20)),
    (   7.19,  -15.0, -math.pi / 2),

    # ── East straight (road(5)): x≈45, going south (heading=-π/2)
    (   7.19,  -25.0, -math.pi / 2),
    (   7.19,  -35.0, -math.pi / 2),

    # ── SE curve (road intersection area, road(10)/(9)): turning south to west
    (   5.5,   -52.0, -math.pi / 2 - math.radians(15)),
    (   1.5,   -60.5, -math.pi),
    (  -5.0,   -65.0, -math.pi + math.radians(10)),

    # ── South inner straight (road(10)): going west
    ( -15.0,   -48.19, math.pi),
    ( -25.0,   -48.19, math.pi),

    # ── SW curve (road(9)): centre (-4.5,-4.5)  – going west then north
    # (inner circuit – skip, outer circuit continues below)

    # ── Outer south straight (road(3)): x≈-64.5, going south (heading=-π/2)
    ( -61.19, -75.0,  -math.pi / 2),
    ( -61.19, -85.0,  -math.pi / 2),
    ( -61.19, -95.0,  -math.pi / 2),
    ( -61.19,-105.0,  -math.pi / 2),
    ( -61.19,-115.0,  -math.pi / 2),
    ( -61.19,-125.0,  -math.pi / 2),
    ( -61.19,-135.0,  -math.pi / 2),

    # ── SW curve (road(2)): centre (-64.5,-64.5), turning south to west
    ( -67.0, -130.0,  -math.pi / 2 - math.radians(15)),
    ( -75.0, -133.5,  -math.pi / 2 - math.radians(35)),
    ( -84.0, -134.8,  -math.pi / 2 - math.radians(60)),
    ( -93.5, -133.5,  -math.pi / 2 - math.radians(80)),
    (-101.0, -129.5,  math.pi - math.radians(5)),

    # ── West straight (road(1) south end) going north back to start
    (-108.69, -120.0,  math.pi / 2),
    (-108.69, -112.0,  math.pi / 2),
]

# Positions where barrels may be spawned (centre of right lane segments)
# Used by the procedural generation in the env reset.
BARREL_SPAWN_CANDIDATES = [
    (-108.69, -70.0),
    (-108.69, -50.0),
    (-108.69, -30.0),
    (-108.69, -10.0),
    ( -45.0,    7.19),
    ( -25.0,    7.19),
    (   7.19,  -20.0),
    (   7.19,  -30.0),
    ( -61.19, -85.0),
    ( -61.19,-105.0),
    ( -61.19,-120.0),
]

# Default (fixed) barrel positions — used for the "fixed obstacles" experiment.
BARREL_FIXED_POSITIONS = [
    (-110.31684, -66.875162, 0.6),
    (-107.04991, -33.037806, 0.6),
    (-103.169,   -54.770003, 0.6),
    ( -45.0,    -105.104,    0.6),
    ( -21.3988,  -45.2699,   0.6),
    ( -22.9161,  -45.3401,   0.6),
    ( -22.1326,  -45.7229,   0.6),
    ( -48.6853,   20.1904,   0.6),
]
