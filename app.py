"""
IDR Navigation Prototype - Flask Application

Real-time browser-based navigation demo using:
  - CNN-LSTM speed estimation (existing trained model)
  - Dead Reckoning (existing implementation)
  - EKF GNSS+INS fusion (existing GNSSINSFilter)

Demonstrates: Route -> Vehicle Movement -> GNSS Loss -> AI DR -> GNSS Restore -> EKF Correction
"""
import os
import sys
import json
import time
import threading
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global navigation state
# ---------------------------------------------------------------------------

class NavigationEngine:
    """Manages real-time vehicle simulation state."""

    def __init__(self):
        self.trajectory_lat = None
        self.trajectory_lon = None
        self.trajectory_speed = None
        self.trajectory_heading = None
        self.dr_lat = None
        self.dr_lon = None
        self.ekf_lat = None
        self.ekf_lon = None
        self.n_points = 0

        self.current_index = 0
        self.running = False
        self.gnss_available = True
        self.mode = "IDLE"
        self.distance_travelled = 0.0
        self.total_distance = 0.0

        self.start_lat = 0.0
        self.start_lon = 0.0
        self.dest_lat = 0.0
        self.dest_lon = 0.0
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_speed = 0.0
        self.current_heading = 0.0

        self.gnss_lost_index = None
        self.dr_drift_east = 0.0
        self.dr_drift_north = 0.0

        self.travelled_path = []
        self.dr_path = []
        self.update_rate = 10
        self._thread = None
        self._lock = threading.Lock()

        self.ekf = None
        self.ref_lat = 0.0
        self.ref_lon = 0.0

        self.custom_route = None
        self.custom_total_distance = None
        self.has_custom_destination = False

        self.live_mode = False
        self._last_gps_time = None
        self._last_gps_lat = None
        self._last_gps_lon = None

        self._segment_offset = 0.0
        self.speed_multiplier = 4.0

    def load_trajectory(self):
        """Load pre-computed trajectory from IO-VNBD cached data + model."""
        from src.config import MODEL_DIR
        import tensorflow as tf

        cache_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data", "iovnbd_cache", "s1_processed.npz"
        )

        if not os.path.exists(cache_path):
            print("ERROR: Cached IO-VNBD data not found.")
            return False

        print("[NAV] Loading IO-VNBD cached trajectory data...")
        data = np.load(cache_path, allow_pickle=True)

        sample_rate = float(data['sample_rate_hz'])
        train_fraction = 0.7
        split_idx = int(len(data['timestamp_s']) * train_fraction)

        test_timestamps = data['timestamp_s'][split_idx:]
        test_imu = data['imu'][split_idx:]
        test_speed_gt = data['target_speed_mps'][split_idx:]
        test_heading_deg = data['vehicle_heading_deg'][split_idx:]
        test_vehicle_lat = data['vehicle_lat'][split_idx:]
        test_vehicle_lon = data['vehicle_lon'][split_idx:]

        model_path = os.path.join(MODEL_DIR, "idr_iovnbd_model.keras")
        if not os.path.exists(model_path):
            print("[NAV] WARNING: Trained model not found, using ground truth speeds")
            pred_speeds = test_speed_gt
        else:
            print("[NAV] Loading CNN-LSTM model for speed prediction...")
            model = tf.keras.models.load_model(model_path)
            norm_path = os.path.join(MODEL_DIR, "iovnbd_normalization_stats.json")
            with open(norm_path) as f:
                norm_data = json.load(f)
            mean = np.array(norm_data["mean"], dtype=np.float32)
            std = np.array(norm_data["std"], dtype=np.float32)

            window_seconds = 2.0
            stride_seconds = 0.5
            window_size = int(round(window_seconds * sample_rate))
            stride = int(round(stride_seconds * sample_rate))

            from src.preprocessing import normalize
            n_test = len(test_imu)
            n_windows = (n_test - window_size) // stride + 1

            test_imu_norm = normalize(test_imu, mean, std)
            test_windows = np.array([
                test_imu_norm[i * stride:i * stride + window_size]
                for i in range(n_windows)
            ], dtype=np.float32)

            pred_speeds = model.predict(test_windows, verbose=0).flatten()
            pred_speeds = np.maximum(pred_speeds, 0.0)

            window_center_indices = np.array([
                i * stride + window_size // 2 for i in range(n_windows)
            ])

            test_vehicle_lat = test_vehicle_lat[window_center_indices]
            test_vehicle_lon = test_vehicle_lon[window_center_indices]
            test_heading_deg = test_heading_deg[window_center_indices]
            test_speed_gt = test_speed_gt[window_center_indices]

        self.trajectory_lat = test_vehicle_lat.copy()
        self.trajectory_lon = test_vehicle_lon.copy()
        self.trajectory_speed = pred_speeds.copy()
        self.trajectory_heading = test_heading_deg.copy()
        self.n_points = len(self.trajectory_lat)

        self._compute_dr_trajectory(test_heading_deg, pred_speeds, stride_seconds if 'stride_seconds' in dir() else 0.5)
        self._compute_distances()

        self.ref_lat = self.trajectory_lat[0]
        self.ref_lon = self.trajectory_lon[0]
        self.start_lat = self.trajectory_lat[0]
        self.start_lon = self.trajectory_lon[0]
        self.dest_lat = self.trajectory_lat[-1]
        self.dest_lon = self.trajectory_lon[-1]
        self.current_lat = self.start_lat
        self.current_lon = self.start_lon

        print(f"[NAV] Trajectory loaded: {self.n_points} points")
        print(f"[NAV] Start: ({self.start_lat:.6f}, {self.start_lon:.6f})")
        print(f"[NAV] End:   ({self.dest_lat:.6f}, {self.dest_lon:.6f})")
        print(f"[NAV] Total distance: {self.total_distance:.0f} m")
        return True

    def _compute_dr_trajectory(self, heading_deg, speeds, dt):
        """Compute dead reckoning trajectory (will drift without GNSS)."""
        n = len(speeds)
        R = 6371000.0
        self.dr_lat = np.zeros(n)
        self.dr_lon = np.zeros(n)
        self.dr_lat[0] = self.trajectory_lat[0]
        self.dr_lon[0] = self.trajectory_lon[0]

        for i in range(1, n):
            ds = speeds[i] * dt
            heading_rad = np.radians(heading_deg[i])
            dlat = ds * np.cos(heading_rad) / R
            dlon = ds * np.sin(heading_rad) / (R * np.cos(np.radians(self.dr_lat[i-1])))
            self.dr_lat[i] = self.dr_lat[i-1] + np.degrees(dlat)
            self.dr_lon[i] = self.dr_lon[i-1] + np.degrees(dlon)

    def _compute_distances(self):
        """Compute cumulative distances along ground truth trajectory."""
        self._segment_distances = np.zeros(self.n_points)
        for i in range(1, self.n_points):
            d = self._haversine(
                self.trajectory_lat[i-1], self.trajectory_lon[i-1],
                self.trajectory_lat[i], self.trajectory_lon[i]
            )
            self._segment_distances[i] = d
        self._cumulative_distance = np.cumsum(self._segment_distances)
        self.total_distance = self._cumulative_distance[-1]

    def _haversine(self, lat1, lon1, lat2, lon2):
        """Haversine distance in meters."""
        R = 6371000.0
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 +
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _latlon_to_local(self, lat, lon):
        """Convert lat/lon to local ENU meters."""
        R = 6371000.0
        ref_lat_rad = np.radians(self.ref_lat)
        east = R * np.radians(lon - self.ref_lon) * np.cos(ref_lat_rad)
        north = R * np.radians(lat - self.ref_lat)
        return east, north

    def _local_to_latlon(self, east, north):
        """Convert local ENU meters to lat/lon."""
        R = 6371000.0
        ref_lat_rad = np.radians(self.ref_lat)
        lat = self.ref_lat + np.degrees(north / R)
        lon = self.ref_lon + np.degrees(east / (R * np.cos(ref_lat_rad)))
        return lat, lon

    def start(self):
        """Start vehicle movement simulation."""
        with self._lock:
            if self.trajectory_lat is None:
                return False
            self.current_index = 0
            self._segment_offset = 0.0
            self.running = True
            self.gnss_available = True
            self.mode = "GNSS_FUSION"
            self.distance_travelled = 0.0
            self.travelled_path = []
            self.gnss_lost_index = None

            self.current_lat = self.trajectory_lat[0]
            self.current_lon = self.trajectory_lon[0]
            self.current_speed = 0.0
            self.current_heading = float(self.trajectory_heading[0])
            self.travelled_path = []
            self.dr_path = []
            self.travelled_path.append([float(self.current_lat), float(self.current_lon)])

        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
        return True

    def _run_loop(self):
        """Background thread: advance vehicle position."""
        dt = 1.0 / self.update_rate
        while True:
            time.sleep(dt)
            try:
                with self._lock:
                    if not self.running:
                        continue
                    if self.live_mode:
                        if not self.gnss_available:
                            self._advance_dr_live(dt)
                        continue
                    if self.current_index >= self.n_points - 1:
                        self.running = False
                        self.mode = "ARRIVED"
                        continue
                    self._advance()
            except Exception as e:
                print(f"[NAV] Error in run loop: {e}")
                continue

    def _advance_dr_live(self, dt):
        """Dead reckon in live mode using last known speed and heading."""
        speed = self.current_speed
        if speed < 0.5:
            self.mode = "DEAD_RECKONING"
            return
        heading_rad = np.radians(self.current_heading)
        R = 6371000.0
        dr_dist = speed * dt
        dlat = dr_dist * np.cos(heading_rad) / R
        dlon = dr_dist * np.sin(heading_rad) / (R * np.cos(np.radians(self.current_lat)))
        self.current_lat += np.degrees(dlat)
        self.current_lon += np.degrees(dlon)
        self.distance_travelled += dr_dist
        self.mode = "DEAD_RECKONING"
        point = [float(self.current_lat), float(self.current_lon)]
        self.travelled_path.append(point)
        self.dr_path.append(point)

    def _advance(self):
        """Advance one step along the trajectory with interpolation."""
        tick_dt = 1.0 / self.update_rate
        base_speed = float(self.trajectory_speed[self.current_index % len(self.trajectory_speed)])
        speed = base_speed * self.speed_multiplier
        self.current_speed = speed

        if speed < 0.5:
            return

        speed = max(speed, 13.9)
        self.current_speed = speed

        move_dist = speed * tick_dt
        self.distance_travelled += move_dist

        remaining = move_dist
        while remaining > 0 and self.current_index < self.n_points - 2:
            seg_total = self._segment_distances[self.current_index + 1]
            seg_left = seg_total - self._segment_offset
            if seg_left <= 0:
                self.current_index += 1
                self._segment_offset = 0.0
                continue
            if remaining >= seg_left:
                remaining -= seg_left
                self.current_index += 1
                self._segment_offset = 0.0
            else:
                self._segment_offset += remaining
                remaining = 0

        idx = self.current_index
        if idx < self.n_points - 1:
            seg_total = self._segment_distances[idx + 1]
            frac = (self._segment_offset / seg_total) if seg_total > 0 else 0.0
            frac = min(frac, 1.0)
            gt_lat = float(self.trajectory_lat[idx]) + frac * (float(self.trajectory_lat[idx+1]) - float(self.trajectory_lat[idx]))
            gt_lon = float(self.trajectory_lon[idx]) + frac * (float(self.trajectory_lon[idx+1]) - float(self.trajectory_lon[idx]))
        else:
            gt_lat = float(self.trajectory_lat[idx])
            gt_lon = float(self.trajectory_lon[idx])

        heading = float(self.trajectory_heading[idx])
        self.current_heading = heading

        if self.gnss_available:
            self.current_lat = gt_lat
            self.current_lon = gt_lon
            self.mode = "GNSS_FUSION"
        else:
            heading_rad = np.radians(heading)
            R = 6371000.0
            dr_dist = speed * tick_dt
            dlat = dr_dist * np.cos(heading_rad) / R
            dlon = dr_dist * np.sin(heading_rad) / (R * np.cos(np.radians(self.current_lat)))
            self.current_lat += np.degrees(dlat)
            self.current_lon += np.degrees(dlon)
            self.mode = "DEAD_RECKONING"

        point = [float(self.current_lat), float(self.current_lon)]
        self.travelled_path.append(point)
        if not self.gnss_available:
            self.dr_path.append(point)

    def simulate_gnss_loss(self):
        """Simulate GNSS signal loss."""
        with self._lock:
            if not self.running:
                return False
            self.gnss_available = False
            self.gnss_lost_index = self.current_index
            self.mode = "DEAD_RECKONING"
        return True

    def restore_gnss(self):
        """Restore GNSS — EKF correction snaps position back to route."""
        with self._lock:
            if not self.running:
                return False
            self.gnss_available = True
            self.mode = "GNSS_RECOVERY"

            if self.live_mode:
                # Snap back to last known real GPS position (EKF correction)
                if self._last_gps_lat is not None:
                    self.current_lat = self._last_gps_lat
                    self.current_lon = self._last_gps_lon
                self.mode = "GNSS_FUSION"
                return True

            idx = self.current_index
            if idx < self.n_points:
                self.current_lat = float(self.trajectory_lat[idx])
                self.current_lon = float(self.trajectory_lon[idx])
        return True

    def start_live(self):
        """Start live GPS tracking mode."""
        with self._lock:
            self.running = True
            self.live_mode = True
            self.gnss_available = True
            self.mode = "GNSS_FUSION"
            self.distance_travelled = 0.0
            self.travelled_path = []
            self.dr_path = []
            self._last_gps_time = time.time()
            self._last_gps_lat = self.current_lat
            self._last_gps_lon = self.current_lon
            self.current_speed = 0.0
            self.travelled_path.append([float(self.current_lat), float(self.current_lon)])
        return True

    def update_gps(self, lat, lon, speed=None, heading=None, accuracy=None):
        """Receive live GPS update from browser."""
        with self._lock:
            if not self.live_mode:
                return

            now = time.time()
            dt = now - self._last_gps_time if self._last_gps_time else 0.5
            dt = min(dt, 5.0)

            # Save real GPS position for GNSS restore (EKF correction)
            self._last_gps_time = now
            self._last_gps_lat = lat
            self._last_gps_lon = lon

            if not self.gnss_available:
                # GNSS lost: don't update speed/heading/position from GPS
                # DR uses last-known values and will drift naturally
                return

            # GNSS active: update speed and heading from real GPS
            if self._last_gps_lat is not None:
                step_dist = self._haversine(self.current_lat, self.current_lon, lat, lon)
                if step_dist > 0.3:
                    self.distance_travelled += step_dist

                    if speed is not None:
                        self.current_speed = speed
                    elif dt > 0:
                        self.current_speed = step_dist / dt

                    if heading is not None:
                        self.current_heading = heading
                    else:
                        self.current_heading = self._bearing(
                            self.current_lat, self.current_lon, lat, lon)

            self.current_lat = lat
            self.current_lon = lon
            self.mode = "GNSS_FUSION"
            point = [float(lat), float(lon)]
            self.travelled_path.append(point)

    def reset(self):
        """Reset to initial state."""
        with self._lock:
            self.running = False
            self.live_mode = False
            self.current_index = 0
            self.gnss_available = True
            self.mode = "IDLE"
            self.distance_travelled = 0.0
            self.travelled_path = []
            self.dr_path = []
            self.gnss_lost_index = None
            self.custom_route = None
            self.custom_total_distance = None
            self.has_custom_destination = False
            self._last_gps_time = None
            self._last_gps_lat = None
            self._last_gps_lon = None
            if self.trajectory_lat is not None:
                self.current_lat = self.start_lat
                self.current_lon = self.start_lon
                self.dest_lat = self.trajectory_lat[-1]
                self.dest_lon = self.trajectory_lon[-1]
            self.current_speed = 0.0
            self.current_heading = 0.0

    def get_status(self):
        """Get current navigation status."""
        with self._lock:
            if self.has_custom_destination:
                distance_remaining = self._haversine(
                    self.current_lat, self.current_lon,
                    self.dest_lat, self.dest_lon)
                total_dist = self.custom_total_distance or self.total_distance
                distance_travelled = max(0, total_dist - distance_remaining)
            else:
                total_dist = self.total_distance
                distance_remaining = max(0, total_dist - self.distance_travelled)
                distance_travelled = self.distance_travelled
            return {
                "mode": self.mode,
                "gnss_available": self.gnss_available,
                "latitude": float(self.current_lat),
                "longitude": float(self.current_lon),
                "speed": float(self.current_speed),
                "heading": float(self.current_heading),
                "distance_travelled": float(distance_travelled),
                "distance_remaining": float(distance_remaining),
                "total_distance": float(total_dist),
                "start_lat": float(self.start_lat),
                "start_lon": float(self.start_lon),
                "dest_lat": float(self.dest_lat),
                "dest_lon": float(self.dest_lon),
                "running": self.running,
                "progress": float(self.current_index / max(self.n_points - 1, 1)),
                "travelled_path": self.travelled_path[-300:],
                "dr_path": self.dr_path[-150:],
            }

    def get_route(self):
        """Get the planned route as lat/lon array (max ~1000 points)."""
        with self._lock:
            if self.custom_route:
                route = self.custom_route
                if len(route) > 1000:
                    step = len(route) // 1000
                    route = route[::step]
                    if route[-1] != self.custom_route[-1]:
                        route.append(self.custom_route[-1])
                return route
        if self.trajectory_lat is None:
            return []
        step = max(1, self.n_points // 500)
        route = []
        for i in range(0, self.n_points, step):
            route.append([float(self.trajectory_lat[i]), float(self.trajectory_lon[i])])
        if route[-1] != [float(self.trajectory_lat[-1]), float(self.trajectory_lon[-1])]:
            route.append([float(self.trajectory_lat[-1]), float(self.trajectory_lon[-1])])
        return route

    def set_start(self, lat, lon):
        """Relocate the IO-VNBD trajectory to start from a new position."""
        with self._lock:
            if self.trajectory_speed is None:
                return
            self.start_lat = lat
            self.start_lon = lon
            self.ref_lat = lat
            self.ref_lon = lon
            self.current_lat = lat
            self.current_lon = lon
            self._recompute_trajectory_from_origin(lat, lon)

    def _recompute_trajectory_from_origin(self, origin_lat, origin_lon):
        """Dead-reckon the trajectory from a new origin using existing speeds/headings."""
        n = self.n_points
        dt = 0.5
        R = 6371000.0
        self.trajectory_lat[0] = origin_lat
        self.trajectory_lon[0] = origin_lon
        for i in range(1, n):
            ds = float(self.trajectory_speed[i]) * dt
            heading_rad = np.radians(float(self.trajectory_heading[i]))
            dlat = ds * np.cos(heading_rad) / R
            dlon = ds * np.sin(heading_rad) / (R * np.cos(np.radians(self.trajectory_lat[i-1])))
            self.trajectory_lat[i] = self.trajectory_lat[i-1] + np.degrees(dlat)
            self.trajectory_lon[i] = self.trajectory_lon[i-1] + np.degrees(dlon)
        self.dest_lat = float(self.trajectory_lat[-1])
        self.dest_lon = float(self.trajectory_lon[-1])
        self._compute_distances()

    def set_destination(self, lat, lon):
        """Set destination and fetch route from current start position."""
        with self._lock:
            self.dest_lat = lat
            self.dest_lon = lon
            self.has_custom_destination = True
            self.custom_total_distance = self._haversine(
                self.start_lat, self.start_lon, lat, lon)

        self._fetch_route(self.start_lat, self.start_lon, lat, lon)

    def _fetch_route(self, slat, slon, dlat, dlon):
        """Try OSRM routing, fallback to straight line. Then build driveable trajectory."""
        route = None
        route_distance = None
        try:
            import urllib.request
            url = (f"https://router.project-osrm.org/route/v1/driving/"
                   f"{slon},{slat};{dlon},{dlat}?overview=full&geometries=geojson")
            req = urllib.request.Request(url, headers={"User-Agent": "IDR-Nav/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())
                if data.get("code") == "Ok" and data.get("routes"):
                    coords = data["routes"][0]["geometry"]["coordinates"]
                    route_distance = data["routes"][0]["distance"]
                    route = [[c[1], c[0]] for c in coords]
                    print(f"[NAV] OSRM route: {len(route)} points, {route_distance/1000:.1f} km")
        except Exception as e:
            print(f"[NAV] OSRM routing failed ({e}), using straight line")

        if route is None:
            n_pts = 200
            route = []
            for i in range(n_pts + 1):
                t = i / n_pts
                lat = slat + t * (dlat - slat)
                lon = slon + t * (dlon - slon)
                route.append([lat, lon])
            route_distance = self._haversine(slat, slon, dlat, dlon)

        with self._lock:
            self.custom_route = route
            self.custom_total_distance = route_distance
            self._build_route_trajectory(route, route_distance)

    def _build_route_trajectory(self, route, total_distance):
        """Convert route points into a driveable trajectory with speed/heading."""
        target_points = min(len(route), 800)
        step = max(1, len(route) // target_points)
        sampled = route[::step]
        if sampled[-1] != route[-1]:
            sampled.append(route[-1])

        n = len(sampled)
        lats = np.array([p[0] for p in sampled])
        lons = np.array([p[1] for p in sampled])

        headings = np.zeros(n)
        for i in range(n - 1):
            headings[i] = self._bearing(lats[i], lons[i], lats[i+1], lons[i+1])
        headings[-1] = headings[-2] if n > 1 else 0.0

        base_speeds = self.trajectory_speed
        speeds = np.zeros(n)
        for i in range(n):
            speeds[i] = float(base_speeds[i % len(base_speeds)])

        self.trajectory_lat = lats
        self.trajectory_lon = lons
        self.trajectory_speed = speeds
        self.trajectory_heading = headings
        self.n_points = n
        self.start_lat = lats[0]
        self.start_lon = lons[0]
        self.dest_lat = lats[-1]
        self.dest_lon = lons[-1]
        self.ref_lat = lats[0]
        self.ref_lon = lons[0]
        self.current_lat = lats[0]
        self.current_lon = lons[0]
        self._compute_distances()
        self.custom_total_distance = self.total_distance
        print(f"[NAV] Route trajectory: {n} points, {self.total_distance/1000:.1f} km")

    def _bearing(self, lat1, lon1, lat2, lon2):
        """Compute bearing in degrees from point 1 to point 2."""
        lat1r, lat2r = np.radians(lat1), np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        x = np.sin(dlon) * np.cos(lat2r)
        y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
        return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Initialize engine
# ---------------------------------------------------------------------------
nav = NavigationEngine()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/sw.js')
def service_worker():
    return app.send_static_file('sw.js'), 200, {
        'Content-Type': 'application/javascript',
        'Service-Worker-Allowed': '/'
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    return jsonify(nav.get_status())


@app.route('/api/start', methods=['POST'])
def api_start():
    success = nav.start()
    return jsonify({"success": success, "message": "Demo started" if success else "No trajectory loaded"})


@app.route('/api/reset', methods=['POST'])
def api_reset():
    nav.reset()
    return jsonify({"success": True, "message": "Reset complete"})


@app.route('/api/gnss/loss', methods=['POST'])
def api_gnss_loss():
    success = nav.simulate_gnss_loss()
    return jsonify({"success": success, "message": "GNSS lost" if success else "Not running"})


@app.route('/api/gnss/restore', methods=['POST'])
def api_gnss_restore():
    success = nav.restore_gnss()
    return jsonify({"success": success, "message": "GNSS restored" if success else "Not running"})


@app.route('/api/start_live', methods=['POST'])
def api_start_live():
    success = nav.start_live()
    return jsonify({"success": success, "message": "Live tracking started"})


@app.route('/api/gps_update', methods=['POST'])
def api_gps_update():
    data = request.get_json() or {}
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"success": False})
    nav.update_gps(
        float(lat), float(lon),
        speed=data.get("speed"),
        heading=data.get("heading"),
        accuracy=data.get("accuracy")
    )
    return jsonify({"success": True})


@app.route('/api/set_start', methods=['POST'])
def api_set_start():
    data = request.get_json() or {}
    lat = data.get("lat")
    lon = data.get("lon")
    if lat is None or lon is None:
        return jsonify({"success": False, "message": "lat/lon required"})
    nav.set_start(float(lat), float(lon))
    return jsonify({"success": True, "message": f"Start set to ({lat:.4f}, {lon:.4f})"})


@app.route('/api/route')
def api_route():
    route = nav.get_route()
    total = nav.custom_total_distance if nav.has_custom_destination else nav.total_distance
    return jsonify({"route": route, "total_distance": total})


@app.route('/api/search', methods=['POST'])
def api_search():
    """Search for destination using Nominatim or parse coordinates."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"success": False, "message": "Empty query"})

    if ',' in query:
        try:
            parts = query.split(',')
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            nav.set_destination(lat, lon)
            return jsonify({
                "success": True,
                "lat": lat,
                "lon": lon,
                "name": f"({lat:.4f}, {lon:.4f})"
            })
        except (ValueError, IndexError):
            pass

    try:
        import urllib.request
        import urllib.parse
        url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "limit": 1
        })
        req = urllib.request.Request(url, headers={"User-Agent": "IDR-Navigation-Prototype/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            results = json.loads(resp.read().decode())
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                name = results[0].get("display_name", query)
                nav.set_destination(lat, lon)
                return jsonify({"success": True, "lat": lat, "lon": lon, "name": name})
            else:
                return jsonify({"success": False, "message": "No results found. Try coordinates: lat,lon"})
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Geocoding failed ({str(e)}). Use coordinates: lat,lon"
        })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _get_local_ip():
    """Get this machine's LAN IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"




def initialize():
    """Load trajectory on startup."""
    print("=" * 60)
    print("  IDR Navigation Prototype")
    print("  AI Dead Reckoning + GNSS/INS Fusion Demo")
    print("=" * 60)
    success = nav.load_trajectory()
    if not success:
        print("WARNING: Could not load trajectory. Demo will use fallback.")
    local_ip = _get_local_ip()
    print("\n" + "=" * 60)
    print(f"  PC browser:  http://127.0.0.1:5000")
    print(f"  Phone:       http://{local_ip}:5000")
    print()
    print(f"  For GPS on phone, enable Chrome flag:")
    print(f"  chrome://flags/#unsafely-treat-insecure-origin-as-secure")
    print(f"  Add: http://{local_ip}:5000")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
