import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

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

#Defining Meshgrid and locating masses at pixel Centres
	
x_start=-1.5
x_end=1.5
y_start=-1.5
y_end=1.5

## Number of pixels i.e., little rectangles facilitating sharp visulaization

n_pixel_x=10
n_pixel_y=10

dx=(x_end-x_start)/n_pixel_x
dy=(y_end-y_start)/n_pixel_y

xv=np.linspace(x_start,x_end,n_pixel_x+1)
yv=np.linspace(y_start,y_end,n_pixel_y+1)

xc=np.linspace(x_start+(dx/2),x_end-(dx/2),n_pixel_x)
yc=np.linspace(y_start+(dy/2),y_end-(dy/2),n_pixel_y)

xv_2d,yv_2d=np.meshgrid(xv,yv)
xc_2d,yc_2d=np.meshgrid(xc,yc)


#Computing Gravitational Force and analyzing through quiver plots

mass = np.array([MASSES[1],MASSES[0]]) # This mass variable is used in computing the acceleration 
for step in tqdm(range(NUMBER_OF_TIME_STEPS+1)):
    #potential_sum = np.zeros_like(xc_2d)
    force_X = np.zeros_like(xc_2d)
    force_Y = np.zeros_like(yc_2d)
    for i in range(number_of_planets):
        x1 = POSITIONS[i,0] - xc_2d
        x1_2 = np.square(x1)
        
        y1 = POSITIONS[i,1] - yc_2d
        y1_2 = np.square(y1)
        distance = x1_2 + y1_2
        #potential_sum += (-GRAVITATIONAL_CONSTANT*MASSES[i])*(np.reciprocal(np.sqrt(distance)))
        force_X+= (-GRAVITATIONAL_CONSTANT*MASSES[i]*1)*(np.reciprocal(np.power((np.sqrt(distance)),3)))*x1
        force_Y+= (-GRAVITATIONAL_CONSTANT*MASSES[i]*1)*(np.reciprocal(np.power((np.sqrt(distance)),3)))*y1

    # plotting every single configuration does not make sense
    if step % PLOT_INTERVAL == 0:
        fig, ax = plt.subplots()
        ax.quiver(xc_2d,yc_2d,force_X,force_Y)
        plt.scatter(POSITIONS[:,0], POSITIONS[:,1])
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title("Force at t = {:8.4f} s".format(step * TIME_STEP))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        output_file_path = Path("forces", "{:016d}.png".format(step))
        output_file_path.parent.mkdir(exist_ok=True)
        fig.savefig(output_file_path)
        plt.close(fig)
    
    distance_vector = POSITIONS[0]-POSITIONS[1]
    distance_vector_length = (np.linalg.norm(distance_vector))  #Computing the Distance vector Length
    acceleration = (GRAVITATIONAL_CONSTANT*mass/distance_vector_length**3)*np.array([-1*distance_vector,distance_vector]) #Taken care of  the Unit Vector along the displacement direction by taking cube of distance
    POSITIONS = POSITIONS + VELOCITIES*TIME_STEP #position update
    VELOCITIES = VELOCITIES + acceleration * TIME_STEP #velocity update
