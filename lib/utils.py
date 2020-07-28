
# helper functions
import math

def get_resultant(accelerations):
    x = math.pow(accelerations["x"], 2)
    y = math.pow(accelerations.get("y"), 2)
    z = math.pow(accelerations.get("z"), 2)

    return math.sqrt(x+y+z)