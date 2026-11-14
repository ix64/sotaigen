"""Independent upper-thigh girth and lateral chord distance on a fixed skeleton.

The station and anchors are model conventions. Targets are fractions of torso length;
measurements use the entire single-leg contour, not a muscle radius or its volume.
"""

import numpy as np

from sotaigen.form.field import Layer, ramp

ARC_DOMAIN = (-0.6, 1.5)
REAR_DOMAIN = (-0.18, 0.18)


def stations(L):
    span = L["crotch"] - L["knee"]
    return np.array(
        [L["crotch"] + 0.05 * L["T"], L["crotch"] - 0.25 * span, L["crotch"] - 0.70 * span]
    )


def section_measures(curves, L):
    outer = np.array([curve[:, 0].max() for curve in curves])
    z = stations(L)
    t = (z[1] - z[0]) / (z[2] - z[0])
    girth = np.linalg.norm(np.roll(curves[1], -1, axis=0) - curves[1], axis=1).sum()
    return {
        "outerArc": float((outer[1] - (1 - t) * outer[0] - t * outer[2]) / L["T"]),
        "thighGirth": float(girth / L["T"]),
    }


def field_sections(body, L, count=192):
    """Trace the FIRST surface exit of rays from each section's interior.

    The upper anchor may be a joined pelvis; only its lateral edge is used.
    The other sections are closed contours of the left leg.
    """
    z = stations(L)
    hip, knee = L["hipJoint"], L["kneeJoint"]
    centers = hip + (z[:, None] - hip[2]) / (knee[2] - hip[2]) * (knee - hip)
    angle = np.arange(count) * 2 * np.pi / count
    directions = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(count)))
    origin = np.repeat(centers, count, axis=0)
    rays = np.tile(directions, (3, 1))
    if np.any(body(origin) >= 0):
        raise ValueError("the thigh's measuring point lies outside the skin")
    steps = np.linspace(0, 0.65 * L["T"], 129)
    probes = origin[:, None, :] + rays[:, None, :] * steps[None, :, None]
    outside = body(probes.reshape(-1, 3)).reshape(-1, len(steps)) > 0
    if not np.all(outside.any(1)):
        raise ValueError("the thigh has no closed section to measure there")
    first = outside.argmax(1)
    low, high = steps[first - 1], steps[first]
    for _ in range(23):
        mid = (low + high) / 2
        inside = body(origin + rays * mid[:, None]) < 0
        low, high = np.where(inside, mid, low), np.where(inside, high, mid)
    return (origin + rays * ((low + high) / 2)[:, None]).reshape(3, count, 3)


def mesh_measures(vertices, triangles, L):
    """Measure the same chord and one-leg girth on delivered triangles."""
    outer = []
    p = vertices[triangles]
    for z in stations(L):
        a = p[(p[:, :, 2].min(1) <= z) & (p[:, :, 2].max(1) > z)]
        b = np.roll(a, -1, axis=1)
        hit = (a[:, :, 2] > z) != (b[:, :, 2] > z)
        a, b = a[hit], b[hit]
        ends = (a + (b - a) * ((z - a[:, 2]) / (b[:, 2] - a[:, 2]))[:, None]).reshape(-1, 2, 3)
        ends = ends[(ends[:, :, 0] > 0).all(1) & (ends[:, :, 0] < 0.42 * L["T"]).all(1)]
        if not len(ends):
            raise ValueError("the thigh's mesh has no section to measure there")
        outer.append(ends[:, :, 0].max())
        if z == stations(L)[1]:
            girth = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1).sum()
    z = stations(L)
    t = (z[1] - z[0]) / (z[2] - z[0])
    return {
        "outerArc": float((outer[1] - (1 - t) * outer[0] - t * outer[2]) / L["T"]),
        "thighGirth": float(girth / L["T"]),
    }


class ThighProfile(Layer):
    def __init__(self, reference, L, *, arc=None):
        super().__init__(reference)
        self.L = L
        self.high, self.station, self.low = stations(L)
        self.curves = field_sections(reference, L)
        self.half = float(max(curve[:, 0].max() for curve in self.curves))
        self.origin = section_measures(self.curves, L)
        self.arc_target = self.origin["outerArc"] if arc is None else arc
        # The girth is the masses' (upperThighGirth is solved for): the arc is shared out around it.
        self.girth_target = self.origin["thighGirth"]
        self._corrections = {}
        # What a target past this body's reach came to instead, by parameter, in shares of the torso.
        self.reached = {}
        self.b = 0.0
        self.a = self._solve(self._arc, self.arc_target, self._arc_domain(self.b), "outerThighArc")
        self.c = self._compensation(self.a, self.b)

    def band(self, z):
        u = np.clip((z - self.low) / (self.high - self.low), 0, 1)
        peak = (self.station - self.low) / (self.high - self.low)
        a = 3 * peak / min(peak, 1 - peak)
        b = 3 * (1 - peak) / min(peak, 1 - peak)
        return (u / peak) ** a * ((1 - u) / (1 - peak)) ** b

    def _axis(self, z):
        hip, knee = self.L["hipJoint"], self.L["kneeJoint"]
        return hip[1] + (z - hip[2]) / (knee[2] - hip[2]) * (knee[1] - hip[1])

    def _depth_band(self, points):
        x = np.abs(points[:, 0])
        return (
            self.band(points[:, 2])
            * ramp(x / (0.06 * self.L["T"]))
            * ramp((1.25 - x / self.half) / 0.2)
        )

    def _rear(self, distance):
        # Zero on the entire anterior half; C2 at the axis and at the rear plateau.
        t = np.clip(distance / (0.10 * self.L["T"]), 0, 1)
        return t**3 * (10 - 15 * t + 6 * t * t)

    def _map(self, points, a, b, c):
        q = np.array(points, dtype=float, copy=True)
        x = np.abs(q[:, 0])
        u = np.clip(x / self.half, 0, 1)
        band = self.band(q[:, 2])
        depth_band = self._depth_band(q) if b or c else None
        if a:
            gate = ramp((u - 0.42) / 0.36)
            e = np.exp(-a * band * gate)
            placed = self.half * u / (u + (1 - u) * e + 1e-30)
            q[:, 0] += np.sign(q[:, 0]) * np.where((x < self.half) & (e != 1), placed - x, 0)
        if b or c:
            distance = q[:, 1] - self._axis(q[:, 2])
            q[:, 1] += distance * np.expm1((b + c * self._rear(distance)) * depth_band)
        return q

    def _forward(self, points, a, b):
        return self._map(points, a, b, self._compensation(a, b))

    def from_reference(self, points):
        return self._map(points, self.a, self.b, self.c)

    def to_reference(self, points):
        q = np.array(points, dtype=float, copy=True)
        active = (q[:, 2] > self.low) & (q[:, 2] < self.high) & (np.abs(q[:, 0]) < 1.25 * self.half)
        v = q[active].copy()
        if not len(v):
            return q
        if self.a:
            target = np.abs(v[:, 0])
            low = np.zeros(len(v))
            high = np.full(len(v), self.half)
            for _ in range(35):
                mid = (low + high) / 2
                trial = v.copy()
                trial[:, 0] = mid
                below = self._map(trial, self.a, 0, 0)[:, 0] < target
                low, high = np.where(below, mid, low), np.where(below, high, mid)
            v[:, 0] = np.where(target < self.half, np.sign(v[:, 0]) * (low + high) / 2, v[:, 0])
        if self.b or self.c:
            axis = self._axis(v[:, 2])
            band = self._depth_band(v)
            target = (v[:, 1] - axis) * np.exp(-self.b * band)
            distance = target.copy()
            if self.c:
                k = self.c * band
                # The rear map is strictly increasing in the supported domain. Newton
                # inverts only this one-dimensional map, without moving the front.
                for _ in range(8):
                    t = np.clip(distance / (0.10 * self.L["T"]), 0, 1)
                    w = self._rear(distance)
                    slope = 30 * t**2 * (1 - t) ** 2
                    e = np.exp(k * w)
                    derivative = e * (1 + k * t * slope)
                    distance -= (distance * e - target) / derivative
            v[:, 1] = axis + distance
        q[active] = v
        return q

    def _arc(self, a):
        curves = [self._map(curve, a, 0, 0) for curve in self.curves]
        return section_measures(curves, self.L)["outerArc"]

    def _raw_girth(self, a, b, c):
        curve = self._map(self.curves[1], a, b, c)
        return float(np.linalg.norm(np.roll(curve, -1, axis=0) - curve, axis=1).sum() / self.L["T"])

    def _feasible(self, a, b):
        target = self._raw_girth(0, b, 0)
        return (
            self._raw_girth(a, b, REAR_DOMAIN[0]) <= target <= self._raw_girth(a, b, REAR_DOMAIN[1])
        )

    def _arc_domain(self, b):
        bounds = []
        for end in ARC_DOMAIN:
            if self._feasible(end, b):
                bounds.append(end)
                continue
            inside, outside = 0.0, end
            for _ in range(24):
                mid = (inside + outside) / 2
                if self._feasible(mid, b):
                    inside = mid
                else:
                    outside = mid
            bounds.append(inside)
        return tuple(bounds)

    def _compensation(self, a, b):
        key = (a, b)
        if key not in self._corrections:
            self._corrections[key] = self._solve(
                lambda c: self._raw_girth(a, b, c),
                self._raw_girth(0, b, 0),
                REAR_DOMAIN,
                "the outer arc's compensation",
            )
        return self._corrections[key]

    def _measure(self, a, b):
        return section_measures([self._forward(curve, a, b) for curve in self.curves], self.L)

    def _solve(self, value, target, domain, label):
        low, high = domain
        floor, ceiling = value(low), value(high)
        if not floor - 1e-9 <= target <= ceiling + 1e-9:
            # Past what this body can reach: the nearer end, and what it measures there.
            end, self.reached[label] = (low, floor) if target < floor else (high, ceiling)
            return end
        if abs(value(0) - target) < 1e-12:
            return 0.0
        for _ in range(28):
            mid = (low + high) / 2
            if value(mid) < target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    def limits(self):
        """The outer arc this body's profile can bend to, around its girth: not a file clamp."""
        return {"outerArc": [self._arc(a) for a in self._arc_domain(self.b)]}
