import numpy as np
import open3d as o3d
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def transform_matrix(theta, d, a, alpha):
    """
    Compute the homogeneous transformation matrix using DH parameters.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

class SpotArmFK:
    def __init__(self, visualize=True):
        # DH parameters for Spot arm (approximate values)
        # [theta, d, a, alpha]
        self.dh_params = [
            [0, 0, 0, -np.pi/2],                # shoulder_roll
            [0, 0, 0.3385, 0],                  # shoulder_pitch
            [-np.pi/2, 0, 0.073, -np.pi/2],     # elbow_pitch
            [-np.pi, 0.4033, 0, -np.pi/2],      # elbow_yaw
            [np.pi, 0, 0, -np.pi/2],            # wrist_pitch
            [np.pi/2, 0.11745, 0, np.pi/2]       # hand_replative pose
        ]
        # Joint limits (as radians)
        self.joint_limits = [
            (-5*np.pi/6, np.pi),        # shoulder_roll
            (-np.pi, np.pi/6),          # shoulder_pitch
            (0, np.pi),                 # elbow_pitch
            (-8*np.pi/9, 8*np.pi/9),    # elbow_yaw
            (-7*np.pi/12, 7*np.pi/12),  # wrist_pitch
            (-8*np.pi/9, 8*np.pi/9)     # wrist_yaw
        ]
        self.z_color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], 
                        [1, 0, 1], [0, 1, 1], [1, 1, 1]]
        
        # Initialize visualization
        if visualize:
            self.init_visualization()
        
    def init_visualization(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='Spot Arm Visualization', width=1024, height=768)

        self.setup_camera()
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.2, 0.2, 0.2])  # Dark gray background
        opt.point_size = 5.0
        opt.line_width = 2.0

    def setup_camera(self):
        """
        Set up the camera view for better visualization
        """
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([1, 1, 1])
        
    def compute_fk(self, base_pose, joint_angles):
        """
        Compute forward kinematics given base pose and joint angles.
        base_pose: 4x4 transformation matrix of the base pose
        joint_angles: list of 6 joint angles in radians
        """        
        # Initialize transformation matrix
        T = base_pose
        transforms = [T]
        
        # Compute forward kinematics
        for i, theta in enumerate(joint_angles):
            dh = self.dh_params[i]
            Ti = transform_matrix(theta + dh[0], dh[1], dh[2], dh[3])
            T = T @ Ti
            transforms.append(T)
        
        return transforms
    
    def compute_ik(self, base_pose, target_hand_pose):
        """
        Compute inverse kinematics for the robot arm given a target hand pose.
        If no solution is found within tolerance, return all zeros.
        """
        # Objective function to minimize distance between FK result and target pose
        def objective(joint_angles):
            current_transforms = self.compute_fk(base_pose, joint_angles)
            current_hand_pose = current_transforms[-1]  # The last frame is the hand pose
            
            # Calculate position and orientation error
            pos_error = np.linalg.norm(current_hand_pose[:3, 3] - target_hand_pose[:3, 3])
            rot_error = np.linalg.norm(current_hand_pose[:3, :3] - target_hand_pose[:3, :3])
            
            # Total error as sum of position and orientation errors
            return pos_error + rot_error
        
        # Initial guess for joint angles (start with zero angles)
        initial_guess = [0] * len(self.dh_params)
        
        # Bounds for each joint based on joint limits
        bounds = self.joint_limits
        
        # Optimization to minimize the objective function
        result = minimize(objective, initial_guess, bounds=bounds, method='SLSQP')
        
        # Check if the optimization was successful and if the error is within a tolerance
        if result.success:  # Adjust tolerance as needed
            return result.x  # Return the found joint angles
        else:
            return [0] * len(self.dh_params)  # Return zeros if no solution is found
        
    def visualize(self, transforms):
        """
        Visualize the robot arm using Open3D with markers for each joint's z-axis.
        """

        assert hasattr(self, 'vis'), "Visualization not initialized. Call init_visualization() first."
        # Clear existing geometry
        self.vis.clear_geometries()

        # Position the world frame at a fixed location far from the arm
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # Adjust size as needed
        world_frame.translate([0.5, 0.5, 0])  # Position it 0.5 units along x and y axes, adjust as needed
        self.vis.add_geometry(world_frame)

        # Create coordinate frames and markers for each joint's z-axis
        for i, T in enumerate(transforms):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            frame.transform(T)
            self.vis.add_geometry(frame)
            
            # Calculate the position for the z-axis marker by offsetting along the z-axis
            z_axis_marker_pos = T[:3, 3] + T[:3, 2] * 0.05  # Offset by 0.05 units along z-axis
            
            # Create a small sphere as a marker
            z_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            z_marker.translate(z_axis_marker_pos)
            color = self.z_color[i] if i < len(self.z_color) else [0.5, 0.5, 0.5]  # Gray for hand frame
            z_marker.paint_uniform_color(color)  # Color for each z-axis marker
            
            # Add the marker to the visualization
            self.vis.add_geometry(z_marker)
        
        # Create links between joints
        for i in range(len(transforms) - 1):
            start = transforms[i][:3, 3]
            end = transforms[i + 1][:3, 3]
            
            # Create line set for links
            points = np.vstack((start, end))
            lines = np.array([[0, 1]])
            colors = np.array([[1, 0, 0]])  # Red links
            
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            
            self.vis.add_geometry(line_set)
        
        # Add a grid for better spatial reference
        grid = self.create_grid(size=1.0, grid_spacing=0.1)
        self.vis.add_geometry(grid)
        
        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def create_grid(self, size=1.0, grid_spacing=0.1):
        """
        Create a reference grid in the XY plane
        """
        # Create grid points
        x = np.arange(-size, size + grid_spacing, grid_spacing)
        y = np.arange(-size, size + grid_spacing, grid_spacing)
        
        points = []
        lines = []
        point_count = 0
        
        # Create vertical and horizontal lines
        for i in x:
            points.extend([[i, -size, 0], [i, size, 0]])
            lines.append([point_count, point_count + 1])
            point_count += 2
            
        for i in y:
            points.extend([[-size, i, 0], [size, i, 0]])
            lines.append([point_count, point_count + 1])
            point_count += 2
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.array(points))
        line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
        
        # Set grid color (light gray)
        colors = [[0.7, 0.7, 0.7] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return line_set
    
    def run_interactive(self, base_pose, joint_angles):
        """
        Run interactive visualization
        """
        transforms = self.compute_fk(base_pose, joint_angles)
        self.visualize(transforms)
        
        print("Interactive visualization window opened.")
        print("Controls:")
        print("  Left click + drag: Rotate")
        print("  Right click + drag: Pan")
        print("  Mouse wheel: Zoom")
        print("Press Ctrl+C in terminal to exit")
        
        try:
            while True:
                self.vis.poll_events()
                self.vis.update_renderer()
        except KeyboardInterrupt:
            print("\nClosing visualization...")
        finally:
            self.close()
    
    def close(self):
        """
        Close the visualization window.
        """
        self.vis.destroy_window()

# Example usage
if __name__ == "__main__":
    # Initialize the FK solver
    spot_arm = SpotArmFK(False)
    
    # Set base pose [x, y, z, roll, pitch, yaw]
    base_pose = [0, 0, 0, 0, 0, 0]
    base_pose_mat = np.eye(4)
    base_pose_mat[:3, :3] = R.from_euler('xyz', base_pose[3:]).as_matrix()
    base_pose_mat[:3, 3] = base_pose[:3]
    joint_angles = [np.pi/6, -np.pi/3, np.pi/6, np.pi/3, np.pi/6, 0]
    hand_pose = np.eye(4)
    hand_pose[0] = np.array([0.625     , 0.54126588, 0.5625    , 0.48101155])
    hand_pose[1] = np.array([-0.21650635,  0.8125    , -0.54126588,  0.33643715])
    hand_pose[2] = np.array([-0.75      ,  0.21650635,  0.625     ,  0.58344812])
    
    # ***** Test FK ********
    # Run interactive visualization
    fk_hand_pose = spot_arm.compute_fk(base_pose_mat, joint_angles)[-1]
    assert np.linalg.norm(fk_hand_pose - hand_pose) < 1e-3, "FK test failed"
    # spot_arm.run_interactive(base_pose_mat, joint_angles)

    # ***** Test IK ********
    # Compute IK
    ik_angles = spot_arm.compute_ik(base_pose_mat, hand_pose)
    # Run interactive visualization
    assert np.linalg.norm(spot_arm.compute_fk(base_pose_mat, ik_angles)[-1] - hand_pose) < 1e-2, "IK test failed"
    # spot_arm.run_interactive(base_pose_mat, ik_angles)