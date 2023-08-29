#####Below is the Python animation code for Fermat point for 5 coordinates.#####
import random
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from IPython.display import HTML

def fitness_function(x1,x2):
  num_point = len(point)
  z = 0 
  for i in range(num_point):
    z = ((point[i][0]-x1)**2+(point[i][1]-x2)**2)**0.5+z 
  return z

def update_velocity(particle,velocity,pbest,gbest,w_min=0.5,max=1.0,c=0.1):
  # Initialise new velocity array
  num_particle = len(particle)
  new_velocity = np.array([0.0 for i in range(num_particle)])
  # Randomly generate r1, r2 and inertia weight from normal distribution
  r1 = random.uniform(0,max)
  r2 = random.uniform(0,max)
  w = random.uniform(w_min,max)
  c1 = c
  c2 = c
  # Calculate new velocity
  for i in range(num_particle):
   new_velocity[i] = w*velocity[i]+c1*r1*(pbest[i]-particle[i])+c2*r2*
   (gbest[i]-particle[i])
  return new_velocity

def update_position(particle,velocity):
  # Move particles by adding velocity
  new_particle = particle + velocity
  return new_particle
  
def pso_2d(population,dimension,position_min,position_max,generation,
fitness_criterion):
  # Initialisation
  # Population
  particles = [[random.uniform(position_min,position_max) for j in range(dimension)] 
  for i in range(population)]
  # Particle's best position
  pbest_position = particles
  # Fitness
  pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
# Index of the best particle
  gbest_index = np.argmin(pbest_fitness)
  # Global best particle position
  gbest_position = pbest_position[gbest_index]
  # Velocity (starting from 0 speed)
  velocity = [[0.0 for j in range(dimension)] for i in range(population)]
    
  # Loop for the number of generation
  for t in range(generation):
    # Stop if the average fitness value reached a predefined success criterion
    if np.average(pbest_fitness) <= fitness_criterion:
      break
    else:
      for n in range(population):
        # Update the velocity of each particle
        velocity[n] = update_velocity(particles[n],velocity[n],pbest_position[n],
        gbest_position)
        # Move the particles to new position
        particles[n] = update_position(particles[n],velocity[n])
    # Calculate the fitness value
    pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
    # Find the index of the best particle
    gbest_index = np.argmin(pbest_fitness)
    # Update the position of the best particle
    gbest_position = pbest_position[gbest_index]
  return [gbest_position,fitness_function(gbest_position[0],gbest_position[1])]


# Define the function to animate the particle swarm optimization
def animate_pso_2d(population,dimension,position_min,position_max,generation,
fitness_criterion):
  # Initialisation
  # Population
  particles = [[random.uniform(position_min,position_max) for j in range(dimension)] 
  for i in range(population)]
  # Particle's best position
  pbest_position = particles
  # Fitness
  pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
  # Index of the best particle
  gbest_index = np.argmin(pbest_fitness)
  # Global best particle position
  gbest_position = pbest_position[gbest_index]
  # Velocity (starting from 0 speed)
  velocity = [[0.0 for j in range(dimension)] for i in range(population)]
  
  # Initialise the figure
  fig = plt.figure()
  ax = plt.axes(xlim=(position_min, position_max), ylim=(position_min, position_max))
  plt.title("Particle Swarm Optimization")
  
  # Plot the initial particles
  x = [p[0] for p in particles]
  y = [p[1] for p in particles]
  scat = ax.scatter(x, y)
  
  # Define the update function for the animation
  def update(frame):
    nonlocal particles, pbest_position, pbest_fitness, gbest_index, gbest_position, 
    velocity, scat
    
    # Update the particles' velocity and position
    for n in range(population):
      velocity[n] = update_velocity(particles[n],velocity[n],pbest_position[n],
      gbest_position)
      particles[n] = update_position(particles[n],velocity[n])
      
    # Update the particles' personal best positions and global best position
    pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
    for n in range(population):
      if pbest_fitness[n] < fitness_function(pbest_position[n][0],
      pbest_position[n][1]):
        pbest_position[n] = particles[n].copy()
    gbest_index = np.argmin(pbest_fitness)
    gbest_position = pbest_position[gbest_index].copy()
    
    # Update the scatter plot
    x = [p[0] for p in particles]
    y = [p[1] for p in particles]
    scat.set_offsets(np.c_[x,y])
    
    return scat,
    
  # Create the animation
  anim = animation.FuncAnimation(fig, update, frames=generation, interval=100, 
  blit=True)
  plt.close()
  
  return anim

# Define the parameters for the PSO
population = 100
dimension = 2
position_min = -1.0
position_max = 5.0
generation = 50
fitness_criterion = 0.01
point = [(2,1),(1,2),(2,2.5),(3,2),(3,1)] 
pso_2d (population,dimension,position_min,position_max,generation,
fitness_criterion)

# Run the animation
anim = animate_pso_2d(population,dimension,position_min,position_max,
generation,fitness_criterion)
HTML(anim.to_jshtml())
