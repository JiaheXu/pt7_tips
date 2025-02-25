import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import gi
import time
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class ImageStreamer(Node):
    def __init__(self):
        super().__init__('image_streamer')

        # Initialize GStreamer
        Gst.init(None)

        # Create a GStreamer pipeline
        # self.pipeline_str = """
        # appsrc name=src format=GST_FORMAT_TIME ! videoconvert ! videoscale ! video/x-raw,format=I420,width=640,height=480 !
        # x264enc speed-preset=ultrafast tune=zerolatency bitrate=50000 ! rtph264pay ! 
        # udpsink host=127.0.0.1 port=5000
        # """
        self.last_time = time.time()
        self.pipeline_str = """
        appsrc name=src format=GST_FORMAT_TIME ! queue max-size-time=0 max-size-bytes=0 max-size-buffers=0 ! video/x-raw,framerate=3/1 ! autovideoconvert ! \
        x264enc byte-stream=true tune=zerolatency speed-preset=ultrafast bitrate=3000 ! \
        h264parse ! rtph264pay config-interval=-1 pt=96 ! queue ! \
        udpsink clients=10.3.1.100:5000 max-bitrate=300000000 sync=false async=false
        """
        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.appsrc = self.pipeline.get_by_name("src")

        self.width = 200
        self.height = 150

        # 320 x 240 3hz -> 1.6Mb
        # 160 x 120 3hz -> 500kb
        # 200 x 150 3hz -> 800kb
        self.appsrc.set_property("caps", Gst.Caps.from_string( ("video/x-raw,format=RGB,width={},height={}").format(self.width, self.height) ))
        self.pipeline.set_state(Gst.State.PLAYING)

        # Image Subscriber
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/spot2/camera/hand/image", self.image_callback, 10
        )

    def image_callback(self, msg):
        
        current_time = time.time()
        
        if(current_time - self.last_time < 0.3):
            return
        self.last_time = current_time

        self.get_logger().info( "in callback")
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS Image to OpenCV
        frame = cv2.resize(frame, (self.width, self.height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_data = frame.tobytes()

        # Create GStreamer buffer and push to pipeline
        buffer = Gst.Buffer.new_allocate(None, len(image_data), None)
        buffer.fill(0, image_data)
        self.appsrc.emit("push-buffer", buffer)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)

def main():
    rclpy.init()
    node = ImageStreamer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
