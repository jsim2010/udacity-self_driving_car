# PID Controller Project

See <https://github.com/udacity/CarND-Extended-Kalman-Filter-Project> for installation instructions.

## Reflection

### Describe the effect each of the P, I, D components had in your implementation.

The P component corrected the car based on the current CTE value. This had the effect of steering the car towards the center. However, if the D component was too low, the car would end up overshooting the center and swerve back and forth until jumping off the road.

The I component corrected any bias in the car sensors. Although I was able to find a working implementation with the I component set to 0, I did find that a small I component seemed to stabilize the car. However, having even a little more than a small I component could cause the car to sharply overshoot the middle.

The D component adjusted the steering based on if the car was moving towards or away from the center. This value served to make sure the car would straighten at the center and make the sharper turns before running off the road.

### Describe how the final hyperparameters were chosen.

I chose to use manual tuning to chose the PID parameters. I first set I and D to 0 and played around with different P values until I found a value that allowed the car to stay on the road through the straight section at the beginning of the circle. Once I had found a good P value, I started the D value at a similar value. Initially this value still caused the car to swerve so I increased this value until the car stopped swerving. The P and D values alone were close to passing the simulation. I found a small I value showed slightly better results and after some small adjustments to P and D, the car was able to complete a lap.
