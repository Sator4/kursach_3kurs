import math

from scipy.integrate import odeint
from matplotlib import pyplot as plt
import numpy
import imageio
import time
numpy.set_printoptions(linewidth=5000, precision=2, suppress=True, threshold=numpy.inf)

# from functions import *

size = 100
frame_number = 100
scale = 5
spread_rad = 5
particle_density = 1
generate_threshold = [5, 5]
vortex_pivot = [0.0, 0.0]

data = [[0 for i in range(size)] for j in range(size)]    #пиксели, отрисовка
particles = []  #одномерный массив частиц, которые плавают
distsum_in_pixel = [[0 for i in range(size)] for j in range(size)] # сумма расстояний
# влияющих пискелей, аналог particles_in_pixel
particles_in_pixel = [[0 for i in range(size)] for j in range(size)]

class particle:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight

def cell_to_coord(i):
    return (i-size/2)/(size/scale)

def coord_to_cell(x):
    return round(x*(size/scale) + size/2)

def initial_distribution(x, y, scale):
    return math.atan(y*20 / scale) + math.pi/2

def flow(x, y):
    r = (x*x + y*y)**0.5
    if r == 0:
        return [0.0, 0.0]
    Vt_r = math.tanh(r) / math.cosh(r) ** 2
    theta = math.pi / 2
    if x != 0:
        theta = math.atan(y / x)
    u = -Vt_r * math.sin(theta) * 0.05
    v = Vt_r * math.cos(theta) * 0.05
    return v, u

    # r = (x*x + y*y)**0.5
    # if r == 0:
    #     return [0.0, 0.0]
    # Vt_r = math.tanh(r) / math.cosh(r)**2
    # dxdt = 0.05 * (-Vt_r * y / r)
    # dydt = 0.05 * (Vt_r * x / r)
    # return [dxdt, dydt]



def particles_to_data(move=True):
    for n in range(len(particles)):
        # if n == 100:
        #     print('sum on', n, 'particle', sum([sum(i) for i in data]))
        if move:
            gradient = flow(particles[n].x, particles[n].y)
            particles[n].x += gradient[0]
            particles[n].y += gradient[1]
        y, x = particles[n].y, particles[n].x
        i, j = coord_to_cell(y), coord_to_cell(x)
        i_min, i_max = max(i - spread_rad, 0), min(i + spread_rad, size)
        j_min, j_max = max(j - spread_rad, 0), min(j + spread_rad, size)
        for i1 in range(i_min, i_max):
            for j1 in range(j_min, j_max):
                y1, x1 = cell_to_coord(i1), cell_to_coord(j1)
                distance = ((y1 - y) ** 2 + (x1 - x) ** 2) ** 0.5
                if distance > spread_rad / size * scale:
                    continue
                distance_coef = 1 / (distance + 1)
                data[i1][j1] += particles[n].weight * distance_coef
                distsum_in_pixel[i1][j1] += distance_coef
                particles_in_pixel[i1][j1] += 1

    # min_distsum = [100, 0, 0, 0]
    for i in range(size):
        for j in range(size):
            if distsum_in_pixel[i][j] > 0:
                data[i][j] /= distsum_in_pixel[i][j]
            if distsum_in_pixel[i][j] < generate_threshold[0] or \
                    particles_in_pixel[i][j] < generate_threshold[1]:
                particles.append(particle(cell_to_coord(j), cell_to_coord(i), data[i][j]))

            # if min_distsum[0] > distsum_in_pixel[i][j]:
            #     min_distsum = [distsum_in_pixel[i][j], particles_in_pixel[i][j], i, j]
            # if min_distsum[1] > particles_in_pixel[i][j]:
            #     min_distsum = [distsum_in_pixel[i][j], particles_in_pixel[i][j], i, j]
    # print('min_distsum', min_distsum)
    print('len(particles)', len(particles))

def just_move():
    for n in range(len(particles)):
        gradient = flow(particles[n].x, particles[n].y)
        particles[n].x += gradient[0]
        particles[n].y += gradient[1]

filenames = []
# for n in range(frame_number+1):
#     filenames.append('out_images\\plot' + str(n) + '.png')

begin_time = time.time()

for i in range(size):    ##############  НАЧАЛО КОДА  ###################
    for j in range(size):
        if i % particle_density != 0 or j % particle_density != 0:
            continue
        x = cell_to_coord(j)
        y = cell_to_coord(i)
        # if i == 30 and j == 30:
        #     weight = 1
        # else:
        #     weight = 0
        weight = initial_distribution(x, y, scale)
        particles.append(particle(x + scale / size, y + scale / size, weight))

# particles_to_data(False)
# data_np = numpy.array(data)
# plt.imsave('out_images\\plot0.png', data_np, cmap='plasma')

end_time = time.time()
print(end_time - begin_time)
total_time = 0

run_till = frame_number
for k in range(frame_number):
    while run_till <= k:
        run_till = int(input("New run_till:"))
    if (k+1) % 10 == 0:
        print(k, 'time elapsed =', total_time)
    begin_time = time.time()
    data = [[0 for i in range(size)] for j in range(size)]
    distsum_in_pixel = [[0 for i in range(size)] for j in range(size)]
    particles_in_pixel = [[0 for i in range(size)] for j in range(size)]
    vortex_pivot[0] += 0.01
    vortex_pivot[1] += 0.01

    if k % 10 == 0:
        filenames.append('out_images\\plot' + str(k) + '.png')
        particles_to_data()
        data_np = numpy.array(data)
        plt.imsave(filenames[-1], data_np, cmap='plasma')
    else:
        just_move()

    # data_np = numpy.array(data)
    # plt.imsave(filenames[k+1], data_np, cmap='plasma')
    end_time = time.time()
    total_time += end_time - begin_time

with imageio.get_writer('out_mov\mov.gif',
                        mode='I', duration=0.33) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

particles_to_data()
data_np = numpy.array(data)
plt.imsave('out_images\\plot' + '_last' + '.png', data_np, cmap='plasma')

plt.figure(figsize=(5,5))
plt.imshow(data_np, cmap='gist_heat')
plt.show()

print(total_time)