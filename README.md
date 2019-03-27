# fv_raspi
Fingervision node with video stream from raspberry pi + picam
-------------------------
## system setup
- Ubuntu 16.04LTS
- ROS kinetic
- Raspberry pi 3B+
- Pi camera v1

## Prerequisites
- First, you need to fire an operating system on raspberry pi, [HERE](https://downloads.ubiquityrobotics.com/pi.html) is a great mirror with ROS integrated. Follow the instruction and perform necessary testing. Don't forget to enable pi camera via:
```sh
sudo raspi-config
```

- Clone raspicam_node ros node from [This Link](https://github.com/UbiquityRobotics/raspicam_node). This node publishes image in compressed JPEG format which is better practice to maintain a high message rate compared to incompressed counterparts. Modify the launch file and  you can search for a camera calibraton file (in the extension of .yaml) to replace inappropriate one. Here we use 640X480 frame size and on 30HZ publishing rate.
Launch the raspicam_node with:
```sh
roslaunch raspicam_node <...640X480_30fps.launch>
```
And you should be able to see the publish image use rqt_imageview or simple rostopic echo/hz/bw to check the data, frequency and bandwidth.

- Setup master and slave machine: Here PC will be master, raspi will be slave. Append following script to you ~/.bashrc. After this, you should be able to launch roscore on you master PC and launch raspicam_node on you raspberry and can communicate via ethernet in local network.
```sh
export ROS_MASTER_URI=http://master_machine_ip:11311
```
## Launching fv_raspi
-First clone this repo and compile it in catkin_ws. launch image processing and motion tracking module:
```sh
rosrun fv_raspi Fv_utils.py
rosrun fv_raspi send_command.py
```
After you launch send_command node, you can input 't', 'r', 'l', 's' in terminal to label 30 frames data to be correspondingly: translational slip, rotational slip, rolling, stable. There are hints prompts.
you can also check the streaming image and displacement field with rqt/imageview utilities.


## Collecting dataset (contact_condition_dataset)
To collect contact motion dataset, you can run Fv_utils.py directly and run:
  ```sh
  rosrun fv_raspi send_command.py
  ```
  it may pop out that directory doesn't exist, you can create an folder named contact_condition_dataset and with inside     folders 0, 1, 2, 3.
  This dataset contains datapoints of 30 frames of 2X30X30 displacement vectors that represent dx and dy displacement of finger contact area. This dataset would be for now used to train and test a slip prediction framework.
  
TO BE CONTINUE...
