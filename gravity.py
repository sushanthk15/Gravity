from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# tqdm is used for a progress bar
from tqdm import tqdm

# parameters
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3 kg^-1 s^-2
POSITIONS = np.array([[-1, 0], [1, 0]])
VELOCITIES = np.array([[0, -1], [0, 1]])
MASSES = [4 / GRAVITATIONAL_CONSTANT, 4 / GRAVITATIONAL_CONSTANT]
TIME_STEP = 0.0001  # s
NUMBER_OF_TIME_STEPS = 1000000
PLOT_INTERVAL = 1000

# derived variables
number_of_planets = len(POSITIONS)
number_of_dimensions = 2

# make sure the number of planets is the same for all quantities
assert len(POSITIONS) == len(VELOCITIES) == len(MASSES)
for position in POSITIONS:
    assert len(position) == number_of_dimensions
for velocity in POSITIONS:
    assert len(velocity) == number_of_dimensions

mass = np.array([MASSES[1],MASSES[0]]) #Inverse mass relation used for computing acceleration
for step in tqdm(range(NUMBER_OF_TIME_STEPS+1)):
    # plotting every single configuration does not make sense
	#if step % PLOT_INTERVAL == 0:
	#	fig, ax = plt.subplots()
	#	x = []
	#	y = []
	#	for position in POSITIONS:
	#		x.append(position[0])
	#		y.append(position[1])
	#	ax.scatter(x, y)
	#	ax.set_aspect("equal")
	#	ax.set_xlim(-1.5, 1.5)
	#	ax.set_ylim(-1.5, 1.5)
	#	ax.set_title("t = {:8.4f} s".format(step * TIME_STEP))
	#	ax.set_xlabel("x")
	#	ax.set_ylabel("y")
	#	output_file_path = Path("positions", "{:016d}.png".format(step))
	#	output_file_path.parent.mkdir(exist_ok=True)
	#	fig.savefig(output_file_path)
	#	plt.close(fig)
	
    # the accelerations for each planet are required to update the velocities
    
    distance_vector = POSITIONS[0]-POSITIONS[1]
    distance_vector_length = (np.linalg.norm(distance_vector))**2
    acceleration = (GRAVITATIONAL_CONSTANT*mass/distance_vector_length)*np.array([-1*distance_vector,distance_vector])
    POSITIONS = POSITIONS + VELOCITIES*TIME_STEP
    VELOCITIES = VELOCITIES + acceleration * TIME_STEP


