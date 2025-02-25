import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

class ImageStreamer(Node):
    def __init__(self):
        super().__init__('image_streamer')

        # Initialize GStreamer
        Gst.init(None)

        # Create a GStreamer pipeline
        self.pipeline_str = """
        appsrc name=src format=GST_FORMAT_TIME ! videoconvert ! videoscale ! video/x-raw,format=I420,width=640,height=480 !
        x264enc speed-preset=ultrafast tune=zerolatency bitrate=500 ! rtph264pay ! 
        udpsink host=127.0.0.1 port=5000
        """

        self.pipeline = Gst.parse_launch(self.pipeline_str)
        self.appsrc = self.pipeline.get_by_name("src")
        self.appsrc.set_property("caps", Gst.Caps.from_string("video/x-raw,format=RGB,width=640,height=480"))
        self.pipeline.set_state(Gst.State.PLAYING)

        # Image Subscriber
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/spot2/camera/hand/image", self.image_callback, 10
        )

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Convert ROS Image to OpenCV
        frame = cv2.resize(frame, (640, 480))
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
