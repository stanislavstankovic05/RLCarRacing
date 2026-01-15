import numpy as np

def car_state(env):
    car = getattr(env.unwrapped, "car", None)
    if car is None:
        return None
    v = car.hull.linearVelocity
    speed = float(np.sqrt(v.x*v.x + v.y*v.y))
    angle = float(car.hull.angle)
    pos = car.hull.position
    x, y = float(pos.x), float(pos.y)
    return angle, speed, x, y

def nearest_track_segment(track, x, y):
    pts = np.array([[t[0], t[1]] for t in track], dtype=np.float32)
    p = np.array([x, y], dtype=np.float32)
    d2 = np.sum((pts - p[None, :])**2, axis=1)
    i = int(np.argmin(d2))
    j = (i + 1) % len(track)
    return pts[i], pts[j]

def lateral_error(env):
    track = getattr(env.unwrapped, "track", None)
    st = car_state(env)
    if track is None or st is None or len(track) < 2:
        return 0.0
    _, _, x, y = st
    a, b = nearest_track_segment(track, x, y)
    p = np.array([x, y], dtype=np.float32)
    ab = b - a
    ap = p - a
    t = float(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8))
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    n = np.array([-ab[1], ab[0]], dtype=np.float32)
    n = n / (float(np.linalg.norm(n)) + 1e-8)
    sign = float(np.sign(np.dot(ap, n)))
    dist = float(np.linalg.norm(p - proj))
    return sign * dist

def angle_to_track(env):
    track = getattr(env.unwrapped, "track", None)
    st = car_state(env)
    if track is None or st is None or len(track) < 2:
        return 0.0
    car_angle, _, x, y = st
    a, b = nearest_track_segment(track, x, y)
    ab = b - a
    track_angle = float(np.arctan2(ab[1], ab[0]))
    diff = track_angle - car_angle
    while diff > np.pi:
        diff -= 2*np.pi
    while diff < -np.pi:
        diff += 2*np.pi
    return float(diff)

def speed(env):
    st = car_state(env)
    if st is None:
        return 0.0
    else:
        return float(st[1])
    
def is_offtrack(env, obs_rgb=None):
    car = getattr(env.unwrapped, "car", None)
    if car is not None and hasattr(car, "on_grass"):
        return bool(car.on_grass)
    if obs_rgb is None:
        return False
    
    h, w, _ = obs_rgb.shape
    patch = obs_rgb[int(h*0.8):, int(w*0.4):int(w*0.6), :]

    g = patch[..., 1].mean()
    r = patch[..., 0].mean()
    b = patch[..., 2].mean()
    return (g > r + 10) and (g > b + 10)

def _extract_tile_v1(obs, config):
    """
    A very simple tile-coding feature extractor.
    It takes a small patch from the green channel of the observation, 
    discretizes it, and returns a tuple representing the state.
    """
    # obs is (height, width, channels)
    # We are using frame_stack=1, so obs is a single frame.
    
    # Take a small patch from the center bottom of the screen
    h, w, _ = obs.shape
    patch = obs[int(h*0.8):, int(w*0.4):int(w*0.6), 0] # Grayscale channel

    # Discretize the patch
    num_tiles = config.get("num_tiles", 8)
    bins = np.linspace(0, 255, num_tiles)
    digitized = np.digitize(patch, bins)
    
    # Flatten and return as a hashable tuple
    return tuple(digitized.flatten())


_EXTRACTORS = {
    "tile_v1": _extract_tile_v1,
}

def extract_features(obs, extractor_name, config):
    if extractor_name not in _EXTRACTORS:
        raise ValueError(f"Unknown feature extractor: {extractor_name}")
    return _EXTRACTORS[extractor_name](obs, config)
