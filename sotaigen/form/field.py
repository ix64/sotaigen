"""Signed-distance primitives and smooth blending, in mm, on arrays of points (N, 3)."""

import numpy as np

# Half-width of the band around the midline where the two sides of a paired mass melt together.
SOFT_MIDLINE = 10.0
# Width of the smooth cut at the wrist, ankle and top of the neck, in mm.
CLIP_BLEND = 4.0


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def frame(z_axis, front=(0.0, -1.0, 0.0)):
    """Rotation whose columns are local x, y, z; local z follows z_axis, local y faces away from front."""
    z = unit(z_axis)
    y = -np.asarray(front, float)
    y = unit(y - z * np.dot(y, z))
    x = np.cross(y, z)
    return np.column_stack((x, y, z))


def tilt_x(angle_deg):
    """Local z leaning towards +Y (the back) by a positive angle."""
    a = np.radians(angle_deg)
    return frame((0.0, np.sin(a), np.cos(a)))


def ramp(t):
    """Smoothstep: 0 below 0, 1 above 1, flat at both ends."""
    u = np.clip(t, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class Ellipsoid:
    """radii are x, y and the upper z; low is the lower z radius. taper narrows only the upper half."""

    def __init__(self, center, radii, rotation=None, taper=0.0, paired=True, low=None):
        self.center = np.asarray(center, float)
        self.radii = np.asarray(radii, float)
        self.rotation = np.eye(3) if rotation is None else rotation
        self.taper = taper
        self.paired = paired
        self.low = self.radii[2] if low is None else float(low)

    def __call__(self, p):
        q = (p - self.center) @ self.rotation
        rx, ry = self.radii[0], self.radii[1]
        rz = np.where(q[:, 2] > 0, self.radii[2], self.low)
        if self.taper:
            # u squared up to the top, then a tangent continuation: a clip here would crease the
            # field along a plane through whatever this mass blends with.
            up = np.maximum(q[:, 2] / self.radii[2], 0.0)
            narrowing = np.where(up < 1.0, up * up, 2.0 - 1.0 / np.maximum(up * up, 1.0))
            scale = 1.0 - self.taper * narrowing
            q = q.copy()
            q[:, 0] /= scale
            q[:, 1] /= scale
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        k0 = np.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2)
        k1 = np.sqrt((x / (rx * rx)) ** 2 + (y / (ry * ry)) ** 2 + (z / (rz * rz)) ** 2)
        return np.where(k1 > 1e-9, k0 * (k0 - 1.0) / np.maximum(k1, 1e-9), -self.radii.min())


class RoundCone:
    """Capsule with two radii; flat squashes the local x, y section to make it non-circular.

    flat_end is the squash at b when it differs from the one at a, eased so the ends stay smooth.
    """

    def __init__(
        self, a, b, r1, r2, flat=(1.0, 1.0), front=(0.0, -1.0, 0.0), paired=True, flat_end=None
    ):
        self.a = np.asarray(a, float)
        b = np.asarray(b, float)
        self.length = np.linalg.norm(b - self.a)
        self.rotation = frame(b - self.a, front)
        self.r1, self.r2 = float(r1), float(r2)
        self.flat = np.array([flat[0], flat[1], 1.0], float)
        self.flat_end = self.flat if flat_end is None else np.array([*flat_end, 1.0], float)
        self.paired = paired

    def __call__(self, p):
        q = (p - self.a) @ self.rotation
        eased = ramp(q[:, 2] / self.length)[:, None]
        flat = self.flat + (self.flat_end - self.flat) * eased
        q = q / flat
        length, r1, r2 = self.length, self.r1, self.r2
        l2 = length * length
        rr = r1 - r2
        a2 = l2 - rr * rr
        y = q[:, 2] * length
        z = y - l2
        x2 = (q[:, 0] ** 2 + q[:, 1] ** 2) * l2 * l2
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        d = (np.sqrt(np.maximum(x2 * a2 / l2, 0.0)) + y * rr) / l2 - r1
        top = np.sign(z) * a2 * z2 > k
        bottom = np.sign(y) * a2 * y2 < k
        d = np.where(bottom, np.sqrt(x2 + y2) / l2 - r1, d)
        d = np.where(top, np.sqrt(x2 + z2) / l2 - r2, d)
        return d * np.minimum(flat[:, 0], flat[:, 1])


def smin(a, b, k):
    """Cubic smooth minimum, C2 so no blend edge shows in shading; k is the blend width in mm, one for
    all the points or one each (then above zero)."""
    if np.ndim(k) == 0 and k <= 0:
        return np.minimum(a, b)
    k = k * 1.5
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * h * k / 6.0


def smax(a, b, k):
    return -smin(-a, -b, k)


def soft_fold(x):
    """|x| rounded at the midline so no crease shows there, and equal to |x| away from it.

    A paired mass therefore sits where its landmark says, unlike hyperbolic rounding, which would
    push everything out by the width of the band.
    """
    return x * np.tanh(x / SOFT_MIDLINE)


class Body:
    """Groups of primitives blended inside, then each group blended into the accumulated body."""

    def __init__(self, groups, clips, carve=()):
        self.groups = groups
        self.clips = clips
        self.carve = carve

    def folded(self, p):
        q = p.copy()
        q[:, 0] = np.abs(q[:, 0])
        return q

    def __call__(self, p):
        q = self.folded(p)
        soft = q.copy()
        soft[:, 0] = soft_fold(q[:, 0])
        total = None
        for group in self.groups:
            field = None
            transform = group.get("paired_coordinates")
            local = q
            paired_points = transform(soft) if transform else soft
            if coordinates := group.get("coordinates"):
                local = coordinates(q)
                paired_points = coordinates(paired_points)
            for prim in group["prims"]:
                d = prim(paired_points if prim.paired else local)
                field = d if field is None else smin(field, d, group["k"])
            for point, normal in group.get("clips", ()):
                field = smax(field, (paired_points - point) @ normal, CLIP_BLEND)
            total = field if total is None else smin(total, field, group["join"])
        for prim, k in self.carve:
            # A blend width may vary over the body: a function of the points, in mm.
            total = smax(total, -prim(q), k(q) if callable(k) else k)
        for point, normal in self.clips:
            total = smax(total, (q - point) @ normal, CLIP_BLEND)
        return total

    def gradient(self, p, step=1.0):
        out = np.zeros_like(p)
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            out[:, axis] = (self(p + shift) - self(p - shift)) / (2 * step)
        return out


class Layer(Body):
    """A local contour on top of another field, its `reference`: that field read through a change of
    coordinates, to_reference.

    The cuts belong to the reference: the head and the wrap take them off the field they are given,
    and it is the reference that cuts. Whatever else a layer is asked and doesn't know itself, such as
    the glutes' peak, it reads through to the layers below. anatomy.LAYERS says how they stack.
    """

    def __init__(self, reference):
        self.reference = reference
        super().__init__(reference.groups, reference.clips, reference.carve)

    def __getattr__(self, name):
        if name == "reference":
            raise AttributeError(name)
        return getattr(self.reference, name)

    @property
    def clips(self):
        return self.reference.clips

    @clips.setter
    def clips(self, clips):
        self.reference.clips = clips

    def to_reference(self, points):
        raise NotImplementedError

    def __call__(self, points):
        return self.reference(self.to_reference(points))


def peeled(field, kinds):
    """(the field under its outermost layers of these kinds, those layers, outermost first)."""
    layers = []
    while isinstance(field, kinds):
        layers.append(field)
        field = field.reference
    return field, layers
