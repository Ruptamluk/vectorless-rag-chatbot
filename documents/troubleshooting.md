# Acme Robotics — R1 Troubleshooting Guide

## Robot Will Not Power On
First confirm the battery is seated and charged. Hold the power button for 5
seconds. If the status LED stays off, the robot is not charging — inspect the
docking contacts for debris and reseat the robot on the dock. If the LED blinks
red three times, the battery pack has failed and should be replaced.

## Robot Stops Frequently or Moves Slowly
This usually means the obstacle-detection field is being triggered. Clean the
lidar window and cameras with a microfiber cloth. Ensure aisle width is at least
1 meter. In shared human zones the R1 intentionally caps its speed at 1.5 m/s —
this is a safety feature, not a fault.

## Robot Loses Its Position
If the R1 reports a localization error, it cannot match its surroundings to the
map. Re-run the mapping routine from Fleet Console after any large change to the
floor layout. Avoid placing two identical-looking aisles next to each other.

## Cannot Connect to Fleet Console
Verify the robot and the Fleet Console server are on the same network and that
port 8443 is open. A flashing blue LED means the robot is searching for the
network. If problems persist, restart the robot and re-pair it from the console.

## Error Codes
- **E101** — battery temperature out of range; let the robot cool or warm before use.
- **E204** — motor stall detected; check wheels for obstructions.
- **E330** — firmware mismatch; update the robot from Fleet Console.
- **E500** — internal fault; contact Acme support with the robot serial number.
