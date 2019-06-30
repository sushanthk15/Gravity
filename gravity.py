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
POSITIONS_STORE = [POSITIONS]
mass = np.array([MASSES[1],MASSES[0]]) #Inverse Mass Relationship
for step in tqdm(range(NUMBER_OF_TIME_STEPS+1)):
    # plotting every single configuration does not make sense
	if step % PLOT_INTERVAL == 0:
		fig, ax = plt.subplots()
		x = []
		y = []
		for position in POSITIONS:
			x.append(position[0])
			y.append(position[1])
		ax.scatter(x, y)
		ax.set_aspect("equal")
		ax.set_xlim(-1.5, 1.5)
		ax.set_ylim(-1.5, 1.5)
		ax.set_title("t = {:8.4f} s".format(step * TIME_STEP))
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		output_file_path = Path("positions", "{:016d}.png".format(step))
		output_file_path.parent.mkdir(exist_ok=True)
		fig.savefig(output_file_path)
		plt.close(fig)
	
    # the accelerations for each planet are required to update the velocities
    
    distance_vector = POSITIONS[0]-POSITIONS[1]
    distance_vector_length = (np.linalg.norm(distance_vector))
    acceleration = (GRAVITATIONAL_CONSTANT*mass/distance_vector_length**3)*np.array([-1*distance_vector,distance_vector]) #Taken care of  the Unit Vector along the displacement direction by taking cube of distance
    #Semi-Implicit Time Integration
    VELOCITIES = VELOCITIES + acceleration * TIME_STEP ##Getting updated velocity by v_new v_old + a_old*delta_t
	POSITIONS = POSITIONS + VELOCITIES*TIME_STEP #getting updated position x_new = x_old + v_new*delta_t
	POSITIONS_STORE.append(POSITIONS)

#converting the POSITIONS list into array to facilitate plotting
position_array = np.array(POSITIONS_STORE) #Shape will be (N,2,2)

## Plotting the Trajectories of Planet 1 and 2

fig, ax = plt.subplots(ncols=2, figsize=(15,5))
for i in range(0, position_array.shape[1]):
    ax[i].plot(position_array[:, i, 0], position_array[:, i, 1]) #Slicing so as to plot X and Y 
    ax[i].set_title(str('Trajectory of Planet:')+str(i+1), fontsize = 20)
    ax[i].set_xlabel("x", fontsize=20)
    ax[i].set_ylabel("y", fontsize = 20)
    fig.savefig("Trajectotres_of_planets_Semi_Implicit_Euler.png")
    plt.close(fig)


