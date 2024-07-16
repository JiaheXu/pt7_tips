
import argparse
import rosbag
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np



def main():
    parser = argparse.ArgumentParser(description="Invert image color from a ROS bag.")
    parser.add_argument("-b", "--bag_in", help="Input ROS bag name.")
    parser.add_argument("-o", "--bag_out", help="Output ROS bag name.")

    parser.add_argument(
        "-t", "--other_topics", nargs="*", default=[], help="Include some other topics."
    )
    args = parser.parse_args()

    print(
        "Invert images from %s into %s..."
        % (args.bag_in, args.bag_out)
    )

    bagIn = rosbag.Bag(args.bag_in, "r")
    bagOut = rosbag.Bag(args.bag_out, "w")
    bridge = CvBridge()

    count = 0
    for topic, msg, t in bagIn.read_messages(topics=["/camera_image0"]):
        cv_img = np.array(bridge.imgmsg_to_cv2(msg))
        #print(cv_img.shape)
        cv_img = cv_img[:,:,0:3]
        #print(cv_img.shape)
        img_msg = bridge.cv2_to_imgmsg(cv_img, encoding="rgb8")
        img_msg.header = msg.header
        bagOut.write("/camera_image0", img_msg, msg.header.stamp )
        count += 1
    print("Done converting " + str(count) + " images")

    for topic, msg, t in bagIn.read_messages(topics=["/imu/data"]):
        bagOut.write("/imu/data", msg, msg.header.stamp)
            

    bagIn.close()
    bagOut.close()


        self.subscription = self.create_subscription(CompressedImage,'/iphone/regular_view/arframe_image/compressed',
        self.image_callback,
        10) #update this to a foxglove node?

        #self.image_queue = deque()

        #self.timer = self.create_timer(0.2, self.image_consumer_callback)

        self.frame_buffer = deque()
        self.buffer_size = 60 

        # If multithreading, uncomment the following line
        #self.image_queue_lock = mp.Lock()

        self.last_timestamp = None
        self.frame_gap_threshold = 0.5  # Set threshold in seconds
        self.frame_count = 0
        self.detection_interval = 30  # Detect face every 30 frames, adjust as needed


    def image_callback(self, msg):
        self.get_logger().info("Received image message")
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
