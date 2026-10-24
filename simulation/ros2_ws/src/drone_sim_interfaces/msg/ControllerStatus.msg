std_msgs/Header header
string state
uint8 traversals_completed
uint8 traversals_total
uint8 current_traversal
string direction
# Reference is for the centre of mass (base_link origin), world frame.
geometry_msgs/Point reference_position
geometry_msgs/Vector3 reference_velocity
# Measured centre-of-mass position (ground truth), world frame.
geometry_msgs/Point position
float64 position_error_norm
# Measured optical-centre height above the floor.
float64 camera_height_m
bool finished
