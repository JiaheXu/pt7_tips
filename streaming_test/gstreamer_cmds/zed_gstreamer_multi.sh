gst-launch-1.0 zedsrc ! timeoverlay ! tee name=split has-chain=true ! \
 queue ! autovideoconvert ! fpsdisplaysink \
 split. ! queue max-size-time=0 max-size-bytes=0 max-size-buffers=0 ! autovideoconvert ! \
 x264enc byte-stream=true tune=zerolatency speed-preset=ultrafast bitrate=3000 ! \
 h264parse ! rtph264pay config-interval=-1 pt=96 ! queue ! \
 multiudpsink clients=10.3.1.100:5000,10.3.1.101:5000 max-bitrate=3000000 sync=false async=false
