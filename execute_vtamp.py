from robot_controller import FrankaOSCController
import time
import numpy as np

controller = FrankaOSCController(
        controller_type="OSC_POSE",
        visualizer=False)

# vtamp reset joints
vtamp_reset = [-1.46929639,  0.19567124,  0.8190383,  -1.63386635, -0.07106318,  1.70830441,  0.28554653]

# joint positions before picking up the mug and placing it in microwave
# change this for other mug positions
target_mug_joints = [-1.55178175,  1.24783466,  0.7027411,  -1.4328308,   0.76210769,  1.94489173,  1.49179101]

# just a function to move mug
def move_mug(pos1, pos2):
    # pos1, pos2: start: [x1, y1, z1], target: [x2, y2, z2]
    controller.move_to(np.array(pos1), use_rot = True, target_rot = np.array([[1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]]))
    controller.gripper_move_to(0.01)
    time.sleep(2)

    controller.move_to(np.array(pos2), use_rot = True, target_rot = np.array([[1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]]))
    controller.gripper_move_to(0.08)
    controller.reset(joint_positions = vtamp_reset)

# the tasks to do
tasks = {
    "open_door": [
        # joint positions when grasping the microwave handle
        [-1.19983605,  1.23307515,  0.88731836, -1.53746075,  0.55186391,  1.6719112,  1.7039934 ],
        # joint positions when the microwave door is open and ready to be released
        [-2.30677675,  0.72036369,  0.85721571, -2.38896967,  1.29110643,  2.07568186,  1.0435157 ],
    ],

    "move_to_mug": [
        # move to safety from behind the microwave
        [-2.04549875,  0.13110096,  0.8547124,  -2.27653889,  0.73427079,  2.96539279,  1.13378737],
        [-1.8782282,  -0.06958966,  0.7723828,  -1.8773017,   0.04423827,  1.82666997, -0.35935782],
        # move to a close enough location near the target mug
        [-1.77360516,  1.19879813,  0.67563912, -1.538672,    0.9787703,  2.0041984,  1.45241623]
    ],

    "put_mug_in": [
        target_mug_joints,
        # lift a little to avoid hitting the microwave base
        [-1.71105565,  1.15418094,  1.07358315, -1.58359159,  0.32711236,  1.76966844,  1.7617448 ],
        # move inside the microwave
        [-1.0919486,   0.80454813,  0.69470468, -2.18191279,  0.84807599,  1.79073728,  1.25595399]
    ],

    "move_to_handle": [
        # safely move the gripper outside the microwave
        [-1.71105565,  1.15418094,  1.07358315, -1.58359159,  0.32711236,  1.76966844,  1.7617448 ],
        vtamp_reset,
        # move the gripper to behind the microwave door
        [-2.04549875,  0.13110096,  0.8547124,  -2.27653889,  0.73427079,  2.96539279,  1.13378737],
    ],

    "close_door": [
        # joint positions to grasp the handle
        [-2.29953026,  0.29414794,  0.70831955, -2.58972329,  1.5485248,   1.94432376,  1.10958152],
        # joint positions when the door is closed
        [-1.20573958,  1.23207458,  0.87596561, -1.46927476,  0.50050663,  1.67493944,  1.66494156]
    ]
}

################################ movements #################################
# Initially, release gripper and reset
controller.gripper_move_to(0.08)
time.sleep(2)
controller.reset(joint_positions = vtamp_reset)

# Then, move away the blocking mugs
move_mug([  0.357, -0.335, 0.085], [0.45348811, -0.5230440, 0.08514559]) # mug target

# Finally, open door, put mug in and close door
for k, joints in tasks.items():
    if k == "open_door" or k == "close_door" or k == "put_mug_in":
        # For these 3 tasks, first close gripper, perform the task, then open gripper
        controller.reset(joint_positions = joints[0])
        controller.gripper_move_to(0.01)
        time.sleep(2)
        if len(joints) > 1:
            for j in joints[1: ]:
                # If we want the robot to move faster, change "duration" to less than 3
                controller.reset(joint_positions = j, duration = 6)
        controller.gripper_move_to(0.08)

        if k == "put_mug_in":
            # This is a Frankapy problem, sometimes the robot does not releast the object and have to wait a long time to do so
            # I will solve this some time though
            controller.gripper_move_to(0.08)
            time.sleep(60)

    else:
        # otherwise just move the robot, no need to care about the gripper
        for j in joints:
            # If we want the robot to move faster, change "duration" to less than 3
            controller.reset(joint_positions = j, duration = 6)

controller.reset(joint_positions = vtamp_reset)   