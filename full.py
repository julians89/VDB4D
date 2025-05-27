import vdbfusion
import numpy as np
from typing import Iterator, Any, Dict, Tuple, List


def load_trajectory(file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load world-frame keyframe poses from a file.
    Returns timestamps (M,) and poses (M,4,4).
    """
    # TODO: implement file parsing (e.g., CSV, JSON, ROS bag)
    raise NotImplementedError


def read_pointcloud(file: str) -> Tuple[np.ndarray, float]:
    """
    Load pointcloud and timestamp from a file path.
    Returns points (N,3) and timestamp.
    """
    raise NotImplementedError


def read_image(file: str) -> Tuple[np.ndarray, float]:
    """
    Load image array and timestamp from a file path.
    Returns image array and timestamp.
    """
    raise NotImplementedError

class PoseDataset:
    """
    Provides time-interpolated world poses and sensor-specific transforms.

    Configuration keys:
      - 'trajectory_file': str path to time-stamped world poses
      - 'calibrations': Dict[str, np.ndarray] mapping sensor name to 4x4 static extrinsic
    """
    def __init__(self, pose_config: Dict[str, Any]):
        ts, poses = load_trajectory(pose_config['trajectory_file'])
        assert ts.ndim == 1 and poses.ndim == 3 and poses.shape[1:] == (4, 4)
        self.key_times = ts
        self.key_poses = poses
        self.calibrations = pose_config.get('calibrations', {})

    def interpolate_world(self, timestamp: float) -> np.ndarray:
        idx = np.searchsorted(self.key_times, timestamp)
        if idx == 0:
            return self.key_poses[0]
        if idx >= len(self.key_times):
            return self.key_poses[-1]
        t0, t1 = self.key_times[idx-1], self.key_times[idx]
        p0, p1 = self.key_poses[idx-1], self.key_poses[idx]
        alpha = (timestamp - t0)/(t1 - t0)
        # Interpolate translation linearly
        trans = (1-alpha)*p0[:3,3] + alpha*p1[:3,3]
        # Keep rotation from nearest keyframe (could use SLERP here)
        rot = p0[:3,:3]
        pose = np.eye(4)
        pose[:3,:3] = rot
        pose[:3,3] = trans
        return pose

    def get_pose(self, sensor: str, timestamp: float) -> np.ndarray:
        world = self.interpolate_world(timestamp)
        extr = self.calibrations.get(sensor, np.eye(4))
        return world @ extr

class LidarScan:
    """LiDAR scan in sensor frame."""
    def __init__(self, points: np.ndarray, pose: np.ndarray, timestamp: float):
        assert points.ndim == 2 and points.shape[1] == 3
        assert pose.shape == (4,4)
        self.points = points
        self.pose = pose
        self.timestamp = timestamp
    def to_world(self) -> np.ndarray:
        N = self.points.shape[0]
        homo = np.hstack((self.points, np.ones((N,1))))
        world = (self.pose @ homo.T).T
        return world[:,:3]

class ImageFrame:
    """Camera image in sensor frame."""
    def __init__(self, image: np.ndarray, pose: np.ndarray, timestamp: float):
        assert pose.shape == (4,4)
        self.image = image
        self.pose = pose
        self.timestamp = timestamp

class LidarDataset:
    """Iterable LiDAR dataset."""
    def __init__(self, config: Dict[str, Any], pose_ds: PoseDataset, sensor_name: str):
        self.files = config['files']
        self.pose_ds = pose_ds
        self.sensor_name = sensor_name
    def __iter__(self) -> Iterator[LidarScan]:
        for f in self.files:
            pts, ts = read_pointcloud(f)
            pose = self.pose_ds.get_pose(self.sensor_name, ts)
            yield LidarScan(pts, pose, ts)

class ImageDataset:
    """Iterable camera dataset."""
    def __init__(self, config: Dict[str, Any], pose_ds: PoseDataset, sensor_name: str):
        self.files = config['files']
        self.pose_ds = pose_ds
        self.sensor_name = sensor_name
    def __iter__(self) -> Iterator[ImageFrame]:
        for f in self.files:
            img, ts = read_image(f)
            pose = self.pose_ds.get_pose(self.sensor_name, ts)
            yield ImageFrame(img, pose, ts)

class UnifiedDataset:
    """
    Holds multiple sensors (LiDARs or cameras) with shared PoseDataset.

    sensor_configs: Dict mapping sensor_name -> { 'type': 'lidar'|'camera', 'config': {...} }
    """
    def __init__(self,
                 sensor_configs: Dict[str, Dict[str, Any]],
                 pose_cfg: Dict[str, Any]):
        self.pose_ds = PoseDataset(pose_cfg)
        self.datasets: Dict[str, Any] = {}
        self.types: Dict[str, str] = {}
        for name, desc in sensor_configs.items():
            stype = desc['type']
            cfg = desc['config']
            self.types[name] = stype
            if stype == 'lidar':
                self.datasets[name] = LidarDataset(cfg, self.pose_ds, name)
            elif stype == 'camera':
                self.datasets[name] = ImageDataset(cfg, self.pose_ds, name)
            else:
                raise ValueError(f"Unknown sensor type {stype}")
    def sensors(self) -> List[str]:
        return list(self.datasets.keys())
    def lidar_sensors(self) -> List[str]:
        return [n for n,t in self.types.items() if t=='lidar']
    def camera_sensors(self) -> List[str]:
        return [n for n,t in self.types.items() if t=='camera']
    def get(self, sensor_name: str) -> Iterator[Any]:
        return iter(self.datasets[sensor_name])

class TSDFReconstructor:
    """Wraps vdbfusion TSDF integration and mesh extraction."""
    def __init__(self, voxel_size: float = 0.05, sdf_trunc: float = 0.3):
        self.voxgrid = vdbfusion.VDBVolume(voxel_size=voxel_size, sdf_trunc=sdf_trunc)
    def integrate_lidar(self, scan: LidarScan):
        pts = scan.to_world()
        self.voxgrid.integrate_scan(pts, scan.timestamp)
    def extract_mesh(self) -> Any:
        return self.voxgrid.extract_surface_mesh()
    def extract_voxels(self) -> Any:
        return self.voxgrid.get_voxels()

class Pipeline:
    """Top-level pipeline handling multiple sensors."""
    def __init__(self, unified_ds: UnifiedDataset):
        self.unified_ds = unified_ds
        self.tsdf = TSDFReconstructor()
        self.mesh = None

    def process_lidar(self):
        for name in self.unified_ds.lidar_sensors():
            for scan in self.unified_ds.get(name):
                self.tsdf.integrate_lidar(scan)

    def process_images(self):
        for name in self.unified_ds.camera_sensors():
            for frame in self.unified_ds.get(name):
                # TODO: ray-trace frame.image using frame.pose onto TSDF or mesh
                pass

    def run(self):
        self.process_lidar()
        self.process_images()
        self.mesh = self.tsdf.extract_mesh()

if __name__ == "__main__":
    # Example multi-sensor config
    sensor_configs = {
        'lidar_top': {
            'type': 'lidar',
            'config': {'files': ['lidar1.pcd', 'lidar2.pcd']}
        },
        'front_cam': {
            'type': 'camera',
            'config': {'files': ['img1.png', 'img2.png']}
        },
        'side_cam': {
            'type': 'camera',
            'config': {'files': ['img3.png', 'img4.png']}
        }
    }
    pose_cfg = {
        'trajectory_file': 'path/to/trajectory',
        'calibrations': {
            'lidar_top': np.eye(4),
            'front_cam': np.eye(4),
            'side_cam': np.eye(4)
        }
    }
    unified = UnifiedDataset(sensor_configs, pose_cfg)
    pipeline = Pipeline(unified)
    pipeline.run()
