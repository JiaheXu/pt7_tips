Tips on payload workspace
===========================

here are the tips for devloping on payloads. when you work with payloads you need to:
* 1. sync the code and time to payload
* 2. enter corresponding docker
* 3. compile your code
* 4. run your code
* 4. close all the docker envs

# install the mmpug system manager in your computer
to install mmpug_ws please follow [this](https://bitbucket.org/castacks/mmpug_ugv/src/develop/docs/getting_started.md) 

# connect to payload (pt-007 for example)
you should connect your computer( a basestaion or whatever) to the payload with an ethernet cable. For the network settings, disable ipv6 and dns in ipv4. Set the remaining params in ipv4 addresses as 10.3.1.20x (x = 0-9) and netmask as 255.255.0.0 and gateway 10.3.0.1

# sync the time of payload (pt-007 for example)
    payload_sync_clock -- mmpug@10.3.1.7 (10.3.1.{$payload_number} for other payloads)

# sync code from your computer/basestation
    mmpug sync -- mmpug@10.3.1.7 (10.3.1.{$payload_number} for other payloads)
this cmd will sync your code to pt-007, you should have your code in mmpug_ws on your computer, to install mmpug_ws please follow [this](https://bitbucket.org/castacks/mmpug_ugv/src/develop/docs/getting_started.md) 

# enter docker
    docker-join.bash -n mmpug_estimation (mmpug_common mmpug_drivers mmpug_sim etc)
this cmd will enter the docker env, there are other docker envs suck as mmpug_common mmpug_driver, you can enter anyone you need.

# compile all the code
    deployer -s local.start local.core.build
deployer is a tool we use to mange the system. -s means execute the commands in sequence.  
local.start means set up the dicker env.
local.core.build will build all the docker env. (mmpug_common, mmpug_drivers, mmpug_estimation, mmpug_sim)

# compile a single package (image_sharing for example)
    docker-join.bash -n mmpug_common
    cd ~/mmpug_ws/src/mmpug_common
    catkin build image_sharing
To compile a single package, the most efficient way is to enter the corresponding docker env, and compile the code.  
docker-join.bash -n {$docker_name} will enter the docker environment.  
after you enter the docker env, go to mmpug_ws/src/mmpug_{env} in the docker env, you need and use catkin build for a normal build progress. 
in the docker env, we cannot use catkin clean cmd, we have to manually delete the previous compiling files(in ~/mmpug_ws/build/).

# after you finish
    mmpug launch --stop 
please run this cmd to close all the docker envs, otherwise it will cause unexpected errors.

# launch things on basestation
just use the same command you launch in payloads.
