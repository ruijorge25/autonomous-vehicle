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
    # ── West straight (x=−105, y: −64.5→4.5, heading north) ─────────────
    (-105.0, -12.0, 0.6),   # B0:  upper, ON centre line
    (-110.0, -45.0, 0.6),   # B1:  lower-mid, 5 m west  (slalom A)
    (-100.0, -60.0, 0.6),   # B2:  near SW end, 5 m east (slalom B)
    # ── North straight (y=45, x: −64.5→4.5, heading east) ───────────────
    ( -38.0,  45.0, 0.6),   # B3:  west portion, ON centre line
    (   0.0,  50.0, 0.6),   # B4:  east portion, 5 m north (slalom)
    # ── East straight (x=45, y: 4.5→−64.5, heading south) ───────────────
    (  45.0,  -8.0, 0.6),   # B5:  upper, ON centre line
    (  50.0, -52.0, 0.6),   # B6:  lower, 5 m east (slalom A)
    (  40.0, -28.0, 0.6),   # B7:  mid,   5 m west (slalom B)
    # ── South straight (y=−105, x: 4.5→−64.5, heading west) ─────────────
    (  -8.0,-105.0, 0.6),   # B8:  east portion, ON centre line
    ( -55.0,-110.0, 0.6),   # B9:  west portion, 5 m south (slalom)
    # ── Curves — blocking the apex, forces correct racing line ───────────
    ( -89.0,  30.0, 0.6),   # B10: NW curve inside apex (dist≈35 m from NW centre)
    (  30.0,  30.0, 0.6),   # B11: NE curve inside apex
    (  30.0, -89.0, 0.6),   # B12: SE curve inside apex
    ( -89.0, -89.0, 0.6),   # B13: SW curve inside apex
]

# ── Secondary circuit (roads 8–15) — where traffic cars circulate ────────────
#
# Exact geometry (from city.wbt CurvedRoadSegment translations = arc centres):
#   Curve  9: centre (-4.5,  -4.5) R=40.5   connects x=-45 vertical to y=-45 horizontal
#   Curve 11: centre (64.5,  -4.5) R=40.5   connects y=-45 horizontal to x=105 vertical
#   Curve 13: centre (64.5,  64.5) R=40.5   connects x=105 vertical to y=105 horizontal
#   Curve 15: centre (-4.5,  64.5) R=40.5   connects y=105 horizontal to x=-45 vertical
#
# Tangent points on arcs (each at a cardinal angle, radius=40.5 from centre):
#   Curve  9: entry (-45, -4.5) @ 180°  →  exit (-4.5, -45) @ 270°
#   Curve 11: entry (64.5,-45)  @ 270°  →  exit (105, -4.5)  @   0°
#   Curve 13: entry (105, 64.5) @   0°  →  exit (64.5, 105)  @  90°
#   Curve 15: entry (-4.5,105)  @  90°  →  exit (-45,  64.5) @ 180°
#
# The two RoadIntersections the circuit crosses:
#   Intersection 17 at (-45,  45): X5 goes east here, traffic goes south
#   Intersection 16 at ( 45, -45): X5 goes south here, traffic goes east
#
# Circuit direction: clockwise from above (south → east → north → west).
#
TRAFFIC_WAYPOINTS = [
    # ── x=-45, heading SOUTH (from curve-15 tangent through intersection 17) ──
    (-45.0,  64.5),   # T0:  curve 15 exit, tangent @ 180°
    (-45.0,  45.0),   # T1:  intersection 17 ← X5 crosses here going east
    (-45.0,  15.0),   # T2:  middle of road 8
    (-45.0,  -4.5),   # T3:  curve 9 entry, tangent @ 180°
    # ── Curve 9: centre (-4.5,-4.5), R=40.5, arc 180°→270° (clockwise) ────────
    (-33.1, -33.1),   # T4:  225° midpoint  (-4.5 + 40.5·cos225°, -4.5 + 40.5·sin225°)
    ( -4.5, -45.0),   # T5:  270° = curve 9 exit, tangent
    # ── y=-45, heading EAST (from curve-9 tangent through intersection 16) ─────
    ( 15.0, -45.0),   # T6:  middle section
    ( 45.0, -45.0),   # T7:  intersection 16 ← X5 crosses here going south
    ( 64.5, -45.0),   # T8:  curve 11 entry, tangent @ 270°
    # ── Curve 11: centre (64.5,-4.5), R=40.5, arc 270°→360° (clockwise) ───────
    ( 93.1, -33.1),   # T9:  315° midpoint  (64.5 + 40.5·cos315°, -4.5 + 40.5·sin315°)
    (105.0,  -4.5),   # T10: 0° = curve 11 exit, tangent / road 12 south start
    # ── x=105, heading NORTH ─────────────────────────────────────────────────────
    (105.0,  30.0),   # T11: midpoint road 12
    (105.0,  64.5),   # T12: curve 13 entry, tangent @ 0°
    # ── Curve 13: centre (64.5,64.5), R=40.5, arc 0°→90° (clockwise) ───────────
    ( 93.1,  93.1),   # T13: 45° midpoint  (64.5 + 40.5·cos45°, 64.5 + 40.5·sin45°)
    ( 64.5, 105.0),   # T14: 90° = curve 13 exit / road 14 east start
    # ── y=105, heading WEST ───────────────────────────────────────────────────────
    ( 30.0, 105.0),   # T15: midpoint road 14
    ( -4.5, 105.0),   # T16: curve 15 entry, tangent @ 90°
    # ── Curve 15: centre (-4.5,64.5), R=40.5, arc 90°→180° (clockwise) ─────────
    (-33.1,  93.1),   # T17: 135° midpoint  (-4.5 + 40.5·cos135°, 64.5 + 40.5·sin135°)
    # → wraps back to T0 = (-45.0, 64.5) @ 180°
]

TRAFFIC_SPEED_MS = 8.0   # m/s  ≈ 29 km/h
NUM_TRAFFIC_CARS = 6
