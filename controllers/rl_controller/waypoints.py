"""
Waypoints for the YELLOW CENTRE LINE of the City world outer circuit.

Derived directly from road segment translations in city.wbt:
  road(1)  StraightRoadSegment  (-105,  4.5)  rotation -π/2  length 69  → west straight
  road(0)  CurvedRoadSegment    (-64.5, 4.5)  radius 40.5               → NW curve
  road(7)  StraightRoadSegment  (-25.5, 45)   length 30                 → north straight
  road(6)  CurvedRoadSegment    (4.5,   4.5)  radius 40.5               → NE curve
  road(5)  StraightRoadSegment  (45,    4.5)  rotation -π/2  length 30  → east straight (short)
  road(4)  CurvedRoadSegment    (4.5,  -64.5) radius 40.5               → SE curve
  road(3)  StraightRoadSegment  (-64.5,-105)  length 69                 → south straight
  road(2)  CurvedRoadSegment    (-64.5,-64.5) radius 40.5               → SW curve

Circuit direction: CLOCKWISE (right-hand traffic):
  West: heading north (π/2)  →  North: heading east (0)  →
  East: heading south (-π/2) →  South: heading west (π)

Curve heading formula (clockwise around arc centre cx,cy,R):
  position:  x = cx + R·cos(θ),  y = cy + R·sin(θ)
  heading:   atan2(−cos θ,  sin θ)

Coordinate system: Webots X = east, Y = north.
"""

import math

# Helper — heading on a clockwise arc at parametric angle θ
def _h(t): return math.atan2(-math.cos(t), math.sin(t))

# (x, y, heading_rad)
WAYPOINTS = [
    # ── West straight  road(1): x=−105, y from −64.5→4.5, heading north ──
    (-105.0, -60.0,  math.pi / 2),
    (-105.0, -50.0,  math.pi / 2),
    (-105.0, -40.0,  math.pi / 2),
    (-105.0, -30.0,  math.pi / 2),
    (-105.0, -20.0,  math.pi / 2),
    (-105.0, -10.0,  math.pi / 2),
    (-105.0,   0.0,  math.pi / 2),

    # ── NW curve  road(0): arc centre (−64.5, 4.5), R=40.5 ──────────────
    # θ goes clockwise from π (entry) to π/2 (exit)
    (-101.9,  20.0,  _h(7*math.pi/8)),   # θ=7π/8
    ( -93.1,  33.1,  _h(3*math.pi/4)),   # θ=3π/4
    ( -80.0,  41.9,  _h(5*math.pi/8)),   # θ=5π/8
    ( -64.5,  45.0,  0.0),               # θ=π/2  exit east

    # ── North straight  road(7): y=45, x from −64.5→4.5, heading east ───
    ( -50.0,  45.0,  0.0),
    ( -35.0,  45.0,  0.0),
    ( -20.0,  45.0,  0.0),
    (  -5.0,  45.0,  0.0),
    (   4.5,  45.0,  0.0),

    # ── NE curve  road(6): arc centre (4.5, 4.5), R=40.5 ────────────────
    # θ goes clockwise from π/2 (entry) to 0 (exit)
    (  20.0,  41.9,  _h(3*math.pi/8)),   # θ=3π/8
    (  33.1,  33.1,  _h(  math.pi/4)),   # θ=π/4
    (  41.9,  20.0,  _h(  math.pi/8)),   # θ=π/8
    (  45.0,   4.5, -math.pi / 2),       # θ=0    exit south

    # ── East straight  road(5): x=45, y from 4.5→−64.5, heading south ───
    (  45.0,  -5.0, -math.pi / 2),
    (  45.0, -15.0, -math.pi / 2),
    (  45.0, -25.0, -math.pi / 2),
    (  45.0, -35.0, -math.pi / 2),
    (  45.0, -50.0, -math.pi / 2),
    (  45.0, -60.0, -math.pi / 2),

    # ── SE curve  road(4): arc centre (4.5, −64.5), R=40.5 ──────────────
    # θ goes clockwise from 0 (entry) to −π/2 (exit)
    (  41.9, -80.0,  _h(-  math.pi/8)),  # θ=−π/8
    (  33.1, -93.1,  _h(-  math.pi/4)),  # θ=−π/4
    (  20.0,-101.9,  _h(-3*math.pi/8)),  # θ=−3π/8
    (   4.5,-105.0,  math.pi),           # θ=−π/2 exit west

    # ── South straight  road(3): y=−105, x from 4.5→−64.5, heading west ─
    ( -10.0,-105.0,  math.pi),
    ( -25.0,-105.0,  math.pi),
    ( -40.0,-105.0,  math.pi),
    ( -55.0,-105.0,  math.pi),
    ( -64.5,-105.0,  math.pi),

    # ── SW curve  road(2): arc centre (−64.5, −64.5), R=40.5 ────────────
    # θ goes clockwise from −π/2 (entry) to −π/=π (exit)
    ( -80.0,-101.9,  _h(-5*math.pi/8)),  # θ=−5π/8
    ( -93.1, -93.1,  _h(-3*math.pi/4)),  # θ=−3π/4
    (-101.9, -80.0,  _h(-7*math.pi/8)),  # θ=−7π/8
    (-105.0, -64.5,  math.pi / 2),       # θ=−π    exit north
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
