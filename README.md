Tips on payload workspace
===========================

here are the tips for devloping on payloads. when you work with payloads you need to:
* 1. sync the code to payload
* 2. enter corresponding docker
* 3. compile your code
* 4. run your code
* 4. close all the docker envs

# connection to pt-007
you should connect your computer to pt-007 with an ethernet cable. For the network settings, disable ipv6 and dns in ipv4. Set the remaining params in ipv4 addresses as 10.3.1.20x (x = 0-9) and netmask as 255.255.0.0 and gateway 10.3.0.1

# sync code from your computer/basestation
    mmpug sync -- mmpug@10.3.1.7 
this cmd will sync your code to pt-007, you should have your code in mmpug_ws on your computer, to install mmpug_ws please follow [this](https://bitbucket.org/castacks/mmpug_ugv/src/develop/docs/getting_started.md) 

# enter docker
    docker-join.bash -n mmpug_estimation 
this cmd will enter the docker env, there are other docker envs suck as mmpug_common mmpug_driver, you can enter anyone you need.

# compile your code
after you enter the docker env, go to mmpug_ugv/src/mmpug_{env} you need and use catkin build for a normal build progress. 
in the docker env, we cannot use catkin clean cmd, we have to manually delete the previous compiling files.

# run everything at once
    mmpug launch --launch super_odom
this cmd will run all the docker envs in mmpug, you can check each modules' status on a tmux terminal.

# after you finish
    mmpug launch --stop 
please run this cmd to close all the docker envs, otherwise it will cause unexpected errors.

# launch things on basestation
just use the same command you launch in payloads.
