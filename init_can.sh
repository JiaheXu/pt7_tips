sudo modprobe peak_usb
sudo ip link set can0 up type can bitrate 1000000
sudo ifconfig can0 txqueuelen 1000
