import os
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from bisect import bisect_left

from datetime import datetime
import time

import numpy as np
from pathlib import Path

import tqdm

def time_to_timestamp(line):
    t = float(time.mktime(datetime.strptime(line.strip()[:-3], "%Y-%m-%d %H:%M:%S.%f").timetuple())) #unix timestamp in seconds
    s = "0." + line.strip().split('.')[-1]

    return t + float(s)

import numpy as np

# WGS84 ellipsoid constants
_a = 6_378_137.0            # Equatorial radius
_e2 = 6.69437999014e-3      # First eccentricity squared

def geodetic_to_ecef(lat, lon, h):
    """
    Convert geodetic coords (deg, deg, m) → ECEF (X,Y,Z in meters).
    """
    φ = np.radians(lat)
    λ = np.radians(lon)
    N = _a / np.sqrt(1 - _e2 * np.sin(φ)**2)
    X = (N + h) * np.cos(φ) * np.cos(λ)
    Y = (N + h) * np.cos(φ) * np.sin(λ)
    Z = (N*(1 - _e2) + h) * np.sin(φ)
    return np.array([X, Y, Z])


class LocalENUConverter:
    def __init__(self, lat0, lon0, h0):
        """
        Define local ENU frame at reference geodetic origin.
        """
        self.lat0 = lat0
        self.lon0 = lon0
        self.h0   = h0
        # origin ECEF
        self._ecef0 = geodetic_to_ecef(lat0, lon0, h0)
        # precompute sines/cosines
        self._sinφ0 = np.sin(np.radians(lat0))
        self._cosφ0 = np.cos(np.radians(lat0))
        self._sinλ0 = np.sin(np.radians(lon0))
        self._cosλ0 = np.cos(np.radians(lon0))

    def to_enu(self, lat, lon, h):
        """
        Convert geodetic → local ENU coordinates (east, north, up).
        """
        ecef = geodetic_to_ecef(lat, lon, h)
        dx, dy, dz = ecef - self._ecef0

        # East
        east =  -self._sinλ0*dx + self._cosλ0*dy
        # North
        north = -self._sinφ0*self._cosλ0*dx \
                - self._sinφ0*self._sinλ0*dy \
                + self._cosφ0*dz
        # Up
        up =  self._cosφ0*self._cosλ0*dx \
            + self._cosφ0*self._sinλ0*dy \
            + self._sinφ0*dz

        return np.array([east, north, up])



class KittiCalibration:
    def __init__(self, calib_dir: Path):
        self.calib_dir = calib_dir
        # parse each file
        self.T_imu_velo = self._load_rigid(calib_dir / "calib_imu_to_velo.txt")
        self.T_velo_cam  = self._load_rigid(calib_dir / "calib_velo_to_cam.txt")
        # camera intrinsics / distortions / rectification / projection
        self.K        = {}   # cam -> 3×3
        self.D        = {}   # cam -> (5,)
        self.R_rect   = {}   # cam -> 3×3
        self.P_rect   = {}   # cam -> 3×4
        self._load_cam_to_cam(calib_dir / "calib_cam_to_cam.txt")

    def _load_rigid(self, fn: Path) -> np.ndarray:
        """Load R: 9 vals and T: 3 vals → 4×4 homogeneous."""
        R_mat = None
        T_vec = None
        for line in open(fn):
            key, rest = line.split(":", 1)
            vals = np.fromstring(rest, sep=" ")
            if key.strip() == "R":
                R_mat = vals.reshape(3, 3)
            elif key.strip() == "T":
                T_vec = vals.reshape(3,)
        assert R_mat is not None and T_vec is not None
        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3]  = T_vec
        return T

    def _load_cam_to_cam(self, fn: Path):
        """Parse K_00, D_00, R_rect_00, P_rect_00, … for cams 0–3."""
        for line in open(fn):
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            vals = np.fromstring(rest, sep=" ")
            # split like "K_00", "D_01", "R_rect_02", "P_rect_03"
            parts = key.strip().split("_")
            if len(parts) < 2:
                continue
            field, cam = parts[0], parts[-1]
            if field == "K":
                self.K[cam]      = vals.reshape(3, 3)
            elif field == "D":
                self.D[cam]      = vals        # distortion coeffs
            elif field == "R" and parts[1] == "rect":
                # key = "R_rect_0i"
                self.R_rect[cam] = vals.reshape(3, 3)
            elif field == "P":
                # key = "P_rect_0i"
                self.P_rect[cam] = vals.reshape(3, 4)

    def get_extrinsic(self, sensor_name: str) -> np.ndarray:
        """
        Return 4×4 world→sensor extrinsic:
          T_world_imu @ [imu→velo] @ [velo→cam0] (if cam),
          or T_world_imu @ [imu→velo] (if 'velodyne' or 'velo').
        """
        # imu→velo
        T = self.T_imu_velo.copy()
        if sensor_name.startswith("cam"):
            # imu→cam0 = imu→velo @ velo→cam
            T = T @ self.T_velo_cam
        return T

    def get_camera_matrices(self, cam: str):
        """
        For camera '00','01',… returns:
           K, D, R_rect, P_rect
        """
        return self.K[cam], self.D[cam], self.R_rect[cam], self.P_rect[cam]

# ---- Pose dataset ----------------------------------------------------------
class KittiPoseDataset:
    def __init__(self, oxts_dir: Path, calib: KittiCalibration):
        """
        oxts_dir: folder containing timestamped .txt OXTS packets
        calib_dir: folder containing calibration files (e.g. sensor extrinsics)
        """

        self.calib = calib

        # load OXTS packets: each line timestamp, lat, lon, alt, roll, pitch, yaw
        self.timestamps = []
        self.poses = []  # 4x4 numpy arrays
        for line in open(oxts_dir / "timestamps.txt"):
            ts = time_to_timestamp(line)
            self.timestamps.append(ts)

        for idx, fname in enumerate(sorted((oxts_dir / "data").glob("*.txt"))):
            if fname.name == "timestamps.txt":
                continue
            data = np.loadtxt(fname)
            lat, lon, alt, roll, pitch, yaw = data[0], data[1], data[2], data[3], data[4], data[5]

            if idx == 0:
                converter = LocalENUConverter(lat, lon, alt)

            rot = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
            trans = converter.to_enu(lat, lon, alt)  # replace with proper projection
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = trans
            self.poses.append(T)
        assert len(self.timestamps) == len(self.poses), "Timestamp/Pose mismatch"


    def get_pose(self, timestamp: float, sensor_name: str) -> np.ndarray:
        """Interpolate world→IMU at `timestamp`, then apply sensor extrinsic."""
        # find bracketing poses
        idx = bisect_left(self.timestamps, timestamp)
        if idx == 0:
            T_world_imu = self.poses[0]
        elif idx >= len(self.timestamps):
            T_world_imu = self.poses[-1]
        else:
            t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
            T0, T1 = self.poses[idx-1], self.poses[idx]
            α = (timestamp - t0) / (t1 - t0)

            # --- rotation via Slerp ---
            key_times = [t0, t1]
            # stack the two rotation matrices
            rots = R.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]]))
            slerp = Slerp(key_times, rots)
            r_interp = slerp(timestamp).as_matrix()  # (3×3)

            # --- translation via linear lerp ---
            p_interp = (1 - α) * T0[:3, 3] + α * T1[:3, 3]

            # build the interpolated IMU pose
            T_world_imu = np.eye(4)
            T_world_imu[:3, :3] = r_interp
            T_world_imu[:3, 3]  = p_interp

        # now append the static sensor extrinsic
        T_world_sensor = T_world_imu @ self.calib.get_extrinsic(sensor_name)
        return T_world_sensor


# ---- Lidar dataset ---------------------------------------------------------
class KittiLidarDataset:
    def __init__(self, velodyne_dir: Path, pose_dataset: KittiPoseDataset, sensor_name="velodyne"):
        """
        velodyne_dir: folder of .bin files (one per scan)
        """
        self.files = sorted(velodyne_dir.glob("*.bin"))
        # KITTI timestamps are in a separate file: load them
        self.timestamps = [time_to_timestamp(l.strip()) for l in open(velodyne_dir.parent / "timestamps.txt")]
        assert len(self.files) == len(self.timestamps)
        self.pose_dataset = pose_dataset
        self.sensor_name = sensor_name

    def __iter__(self):
        for path, ts in zip(self.files, self.timestamps):
            # load Nx4 float32 (x,y,z,reflectance)
            scan = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)[:, :3]
            pose = self.pose_dataset.get_pose(ts, self.sensor_name)
            yield {"points": scan, "timestamp": ts, "pose": pose}


# ---- Image dataset ---------------------------------------------------------
import numpy as np
from pathlib import Path
from PIL import Image

class KittiImageDataset:
    def __init__(
        self,
        image_dir: Path,
        pose_dataset: KittiPoseDataset,
        sensor_name: str,
        intrinsic: np.ndarray,
        distortion: np.ndarray,
        rect_R: np.ndarray,
        proj_P: np.ndarray,
    ):
        """
        image_dir: folder of .png/.jpg images
        intrinsic:    3×3 camera matrix K
        distortion:   distortion coeffs D (length 5)
        rect_R:       3×3 rectification rotation R_rect
        proj_P:       3×4 projection matrix P_rect
        """
        self.files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        # assumes e.g. `<root>/cam00_timestamps.txt`
        ts_file = image_dir.parent / "timestamps.txt"
        self.timestamps = [time_to_timestamp(l.strip()) for l in open(ts_file)]
        assert len(self.files) == len(self.timestamps), \
            f"{sensor_name}: #images ({len(self.files)}) != #timestamps ({len(self.timestamps)})"
        
        self.pose_dataset = pose_dataset
        self.sensor_name = sensor_name
        self.intrinsic  = intrinsic
        self.distortion = distortion
        self.rect_R     = rect_R
        self.proj_P     = proj_P

    def __iter__(self):
        for path, ts in zip(self.files, self.timestamps):
            img  = np.array(Image.open(path))
            pose = self.pose_dataset.get_pose(ts, self.sensor_name)
            yield {
                "image":      img,
                "timestamp":  ts,
                "pose":       pose,
                "intrinsic":  self.intrinsic,
                "distortion": self.distortion,
                "rect_R":     self.rect_R,
                "proj_P":     self.proj_P,
            }


# ---- Usage in UnifiedDataset -----------------------------------------------
from typing import Dict

class KittiUnifiedDataset:
    def __init__(self, root: Path, calib_dir: Path):
        self.pose_ds = KittiPoseDataset(root / "oxts", calib_dir)

        # Velodyne
        self.lidar_ds = KittiLidarDataset(root / "velodyne_points", self.pose_ds)

        # Cameras (example for cam0 and cam1)
        # load intrinsics from calibration file
        cam0_K = np.loadtxt(calib_dir / "cam0_intrinsic.txt").reshape(3, 3)
        cam1_K = np.loadtxt(calib_dir / "cam1_intrinsic.txt").reshape(3, 3)
        self.image_ds = {
            "cam0": KittiImageDataset(root / "image_00", self.pose_ds, "cam0", cam0_K),
            "cam1": KittiImageDataset(root / "image_01", self.pose_ds, "cam1", cam1_K),
        }

    def lidar(self):
        return iter(self.lidar_ds)

    def images(self):
        return ((name, sample) for name, ds in self.image_ds.items() for sample in ds)



class KittiUnifiedDataset:
    def __init__(self, root: Path, calib_dir: Path):
        # 1) load all calib files at once
        self.calib = KittiCalibration(calib_dir)

        # 2) pose dataset still reads OXTS, but now uses calib.get_extrinsic()
        self.pose_ds = KittiPoseDataset(root / "oxts", self.calib)

        # 3) LiDAR: velodyne uses imu→velo extrinsic
        self.lidar_ds = KittiLidarDataset(
            root / "velodyne_points/data",
            self.pose_ds,
            sensor_name="velodyne",
        )

        # 4) Cameras: for each cam folder pick up intrinsics + extrinsics
        self.image_ds = {}
        for cam in ["00", "01", "02", "03"]:
            img_dir = root / f"image_{cam}/data"
            K, D, R_rect, P_rect = self.calib.get_camera_matrices(cam)
            self.image_ds[f"cam{cam}"] = KittiImageDataset(
                img_dir,
                self.pose_ds,
                sensor_name=f"cam{cam}",
                intrinsic=K,
                distortion=D,
                rect_R=R_rect,
                proj_P=P_rect,
            )

    def lidar(self):
        return iter(self.lidar_ds)

    def images(self):
        for name, ds in self.image_ds.items():
            for sample in ds:
                yield name, sample



if __name__ == "__main__":
    import os
    import open3d as o3d
    import vdbfusion


    dir_path = os.path.dirname(os.path.realpath(__file__))

    # point this at your local KITTI sequence
    root_dir  = Path(os.path.join(dir_path, "data", "kitti", "2011_09_26_drive_0001_sync"))
    calib_dir = Path(os.path.join(dir_path, "data", "kitti"))

    # create the unified dataset
    dataset = KittiUnifiedDataset(root_dir, calib_dir)


    """
    vdb_volume = vdbfusion.VDBVolume(0.25, 1.)

    for scan in tqdm.tqdm(dataset.lidar()):
        pts = scan["points"].astype(np.float64)               # Nx3 array in sensor frame
        T = scan["pose"]                   # 4x4 world→sensor transform
        vdb_volume.integrate(pts, T)

    # Extract triangle mesh (numpy arrays)
    vert, tri = vdb_volume.extract_triangle_mesh()

    # Visualize the results
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vert),
        o3d.utility.Vector3iVector(tri),
    )

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
    """
    
    # Accumulate all lidar points in world frame
    all_points = []
    for scan in tqdm.tqdm(dataset.lidar()):
        pts = scan["points"]               # Nx3 array in sensor frame
        T = scan["pose"]                   # 4x4 world→sensor transform

        print (scan["timestamp"])
        print (T)

        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Nx4
        pts_world = (T @ pts_h.T).T[:, :3]                   # Nx3
        all_points.append(pts_world)

    all_points = np.vstack(all_points)

    # Create Open3D point cloud and downsample for speed
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points))
    pcd = pcd.voxel_down_sample(voxel_size=0.1)

    # Visualize
    o3d.visualization.draw_geometries([pcd])
    