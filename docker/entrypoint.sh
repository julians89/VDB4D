#!/bin/bash
# Basic entrypoint for ROS Docker containers

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash
echo "Sourced ROS 2 ${ROS_DISTRO}"

# Source the base workspace, if built
if [ -f /ros_ws/install/setup.bash ]
then
  source /ros_ws/install/setup.bash
  echo "Sourced ROS2 base workspace"
fi

# Source the overlay workspace, if built
if [ -f /overlay_ws/install/setup.bash ]
then
  source /overlay_ws/install/setup.bash
  echo "Sourced autonomy overlay workspace"
fi

# Execute the command passed into this entrypoint
exec "$@"