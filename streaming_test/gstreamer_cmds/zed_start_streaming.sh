gst-launch-1.0 zedsrc camera-resolution=5 camera-fps=15 camera-image-flip=2 ! videoscale ! "video/x-raw,height=300,width=400" ! timeoverlay ! queue max-size-time=0 max-size-bytes=0 max-size-buffers=0 !  autovideoconvert ! \
 x264enc byte-stream=true tune=zerolatency speed-preset=ultrafast bitrate=500 ! \
 h264parse ! rtph264pay config-interval=-1 pt=96 ! queue ! \
 udpsink clients=10.3.1.94:5000 max-bitrate=500000 sync=false async=false

