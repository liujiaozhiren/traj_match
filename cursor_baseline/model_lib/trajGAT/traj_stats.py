"""Trajectory lon/lat statistics only (extracted from pre_proc; avoids model.handler import)."""

from __future__ import annotations

import numpy as np


def traj_statistics(traj_data):
    lats, lons = [], []
    for traj in traj_data:
        for p in traj:
            lon, lat = p[0], p[1]
            assert len(p) == 2
            lats.append(lat)
            lons.append(lon)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    x_range, lon_mean, lon_std = (lon_min, lon_max), lons.mean(), lons.std()
    y_range, lat_mean, lat_std = (lat_min, lat_max), lats.mean(), lats.std()
    data_features = (lon_mean, lon_std, lat_mean, lat_std)
    return {"x_range": x_range, "y_range": y_range, "data_features": data_features}
