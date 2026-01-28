import numpy as np
import math


class AircraftStateNormalizer:
    def __init__(self, num_intruders=5):
        self.num_intruders = num_intruders

        self.speed_range = (200, 300)
        self.altitude_range = (20000, 21000)

        self.rel_pos_scale = 0.3
        self.rel_alt_scale = 1000.0
        self.rel_speed_scale = 50.0

        self.max_cross_track_error = 20.0

        self.lat_range = (-90, 90)
        self.lon_range = (-180, 180)

        self.max_sensing_dist_km = 50.0

    def normalize_speed(self, speed):
        return 2 * (speed - self.speed_range[0]) / (self.speed_range[1] - self.speed_range[0]) - 1

    def normalize_altitude(self, alt):
        return 2 * (alt - self.altitude_range[0]) / (self.altitude_range[1] - self.altitude_range[0]) - 1

    def normalize_heading(self, heading):
        heading_rad = np.radians(heading)
        return np.sin(heading_rad), np.cos(heading_rad)

    def normalize_position(self, lat, lon, ref_lon=None, ref_lat=None):
        if ref_lon is not None and ref_lat is not None:
            norm_lat = (lat - ref_lat) / 100000
            norm_lon = (lon - ref_lon) / 100000
        else:
            norm_lat = 2 * (lat - self.lat_range[0]) / (self.lat_range[1] - self.lat_range[0]) - 1
            norm_lon = 2 * (lon - self.lon_range[0]) / (self.lon_range[1] - self.lon_range[0]) - 1

        return norm_lat, norm_lon

    def _project_to_route(self, curr_lat, curr_lon, start_lat, start_lon, end_lat, end_lon):
        route_dist = self._haversine(start_lat, start_lon, end_lat, end_lon)
        route_heading = self._calculate_bearing(start_lat, start_lon, end_lat, end_lon)
        dist_from_start = self._haversine(start_lat, start_lon, curr_lat, curr_lon)

        if route_dist < 1e-3: return 1.0, 0.0, 0.0

        bearing_from_start = self._calculate_bearing(start_lat, start_lon, curr_lat, curr_lon)
        angle_diff = math.radians(bearing_from_start - route_heading)

        along_track_dist = dist_from_start * math.cos(angle_diff)
        cross_track_error = dist_from_start * math.sin(angle_diff)
        progress = along_track_dist / route_dist

        return progress, cross_track_error, route_heading

    def _calculate_egocentric_goal(self, lat, lon, hdg, goal_lat, goal_lon):
        d_lat_km = (goal_lat - lat) * 111.0
        avg_lat_rad = np.radians((lat + goal_lat) / 2.0)
        d_lon_km = (goal_lon - lon) * 111.0 * np.cos(avg_lat_rad)

        hdg_rad = np.radians(hdg)
        cos_h = np.cos(hdg_rad)
        sin_h = np.sin(hdg_rad)

        x_ego = d_lat_km * cos_h + d_lon_km * sin_h
        y_ego = d_lon_km * cos_h - d_lat_km * sin_h

        norm_scale = np.log(1 + self.max_sensing_dist_km)

        x_norm = np.sign(x_ego) * np.log(1 + abs(x_ego)) / norm_scale
        y_norm = np.sign(y_ego) * np.log(1 + abs(y_ego)) / norm_scale

        x_norm = np.clip(x_norm, -1.0, 1.0)
        y_norm = np.clip(y_norm, -1.0, 1.0)

        return x_norm, y_norm

    def normalize_own_state_with_route(self, state, route_info, target_alt, coll_flag):
        lat, lon, spd, alt, hdg = state[0], state[1], state[2], state[3], state[4]
        s_lat, s_lon, e_lat, e_lon = route_info

        progress, xtk_km, route_hdg = self._project_to_route(lat, lon, s_lat, s_lon, e_lat, e_lon)

        f = []
        f.extend(self.normalize_position(lat, lon))
        f.append(self.normalize_speed(spd))
        f.append(self.normalize_altitude(alt))
        f.extend(self.normalize_heading(hdg))

        f.extend(self.normalize_position(s_lat, s_lon))
        f.extend(self.normalize_position(e_lat, e_lon))

        f.append(np.clip(progress, -0.5, 1.5))
        f.append(np.clip(xtk_km / self.max_cross_track_error, -1.0, 1.0))

        f.extend(self.normalize_heading(state[7]))
        f.extend(self.normalize_heading(state[8]))

        alt_diff = alt - target_alt
        f.append(np.clip(alt_diff / 3000.0, -1.0, 1.0))
        f.append(coll_flag)

        ego_gx, ego_gy = self._calculate_egocentric_goal(lat, lon, hdg, e_lat, e_lon)
        f.append(ego_gx)
        f.append(ego_gy)

        while len(f) < 20: f.append(0.0)
        return f[:20], route_hdg

    def normalize_other_aircraft(self, own_id, neighbors_info, own_raw_state, route_heading_ref):
        target_dim = self.num_intruders * 15
        features = []
        ego_lat, ego_lon = own_raw_state[0], own_raw_state[1]

        try:
            own_val = int(own_id.replace('KL', ''))
        except:
            own_val = 0

        for n in neighbors_info:
            f = []
            n_lat, n_lon, n_spd, n_alt, n_hdg = n[0], n[1], n[2], n[3], n[4]

            avg_lat_rad = np.radians((ego_lat + n_lat) / 2)
            d_lat_km = (n_lat - ego_lat) * 111.0
            d_lon_km = (n_lon - ego_lon) * 111.0 * np.cos(avg_lat_rad)
            dist = np.sqrt(d_lat_km ** 2 + d_lon_km ** 2)
            bearing_to_neigh = math.degrees(math.atan2(d_lon_km, d_lat_km))

            rel_angle = bearing_to_neigh - route_heading_ref
            rel_rad = np.radians(rel_angle)

            x_rel_km = dist * math.cos(rel_rad)
            y_rel_km = dist * math.sin(rel_rad)

            f_x = np.clip(x_rel_km / (self.rel_pos_scale * 111.0), -3, 3)
            f_y = np.clip(y_rel_km / (self.rel_pos_scale * 111.0), -3, 3)

            f.extend(self.normalize_position(n_lat, n_lon))
            f.append(self.normalize_speed(n_spd))
            f.append(self.normalize_altitude(n_alt))
            f.extend(self.normalize_heading(n_hdg))

            if len(n) > 6:
                flag_val = n[6]
            else:
                flag_val = 0.0

            f.append(flag_val)

            try:
                other_val = int(n[5].replace('KL', ''))
            except:
                other_val = 0
            id_rank = 1.0 if own_val > other_val else -1.0
            f.append(id_rank)

            f.extend([f_x, f_y])
            hdg_pos = self._calculate_bearing(ego_lat, ego_lon, n_lat, n_lon)
            f.extend(self.normalize_heading(hdg_pos))

            while len(f) < 15: f.append(0.0)
            features.extend(f)

        if len(features) < target_dim: features += [0.0] * (target_dim - len(features))
        return features[:target_dim]

    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
            dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    @staticmethod
    def _calculate_bearing(lat1, lon1, lat2, lon2):
        lat1, lat2, dlon = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        return (math.degrees(math.atan2(y, x)) + 360) % 360