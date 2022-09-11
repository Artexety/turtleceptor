# Turtleceptor :turtle:
POC of our monocular obstacle detection and avoidance system based on Deep Reinforcement Learning. This project was created as an entry for the German BWKI competition using the infamous turtlebot2 ros package. The goal of this project was to develop a system that would allow the turtlebot to move freely without colliding with any obstacles.  


## Getting Started
- setup ROS Kinetic for Ubuntu 16.04
- setup python venv for Python 3.5.2 using `python3 -m venv`
- install turtlebot2 ros package following this [Tutorial](https://www.youtube.com/watch?v=pDps6eRyPWk)
- install requirements.txt using `pip3 install -r`


## Training the Network
When everything is set up, you can run the commands below to start training. Make sure you `source` your python virtual environment beforehand.

```console
$ roslaunch turtlebot_gazebo turtlebot_world.launch \ 
            world_file:=/path/to/train_world.world

$ python3 train.py
```

## Testing the Network
To test the trained network just run the following commands:
```console
$ roslaunch turtlebot_gazebo turtlebot_world.launch \
            world_file:=/path/to/test_world.world

$ python3 validate.py
```