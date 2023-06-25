import math

from matplotlib import pyplot as plt
import numpy
import imageio
import time
numpy.set_printoptions(linewidth=5000, precision=2, suppress=True, threshold=numpy.inf)

size = 400
frame_number = 500
scale = 5
spread_rad = 1
particle_density = 10
generate_threshold = [0, 0]
speed = 0.05
interpolation_type = 0 # 0 - no, 1 - simplest, 2 - with weights
write_output = True
pallete = 'gist_heat'


data = [[0 for i in range(size)] for j in range(size)]    #пиксели, отрисовка
particles = []  #одномерный массив частиц, которые плавают
distsum_in_pixel = [[0 for i in range(size)] for j in range(size)] # сумма расстояний
# влияющих пискелей, аналог particles_in_pixel
particles_in_pixel = [[0 for i in range(size)] for j in range(size)]
filenames = []
flow_grid = [[0 for i in range(size*3)] for j in range(size*3)]
watchlist = [4690, 3152, 1787, 3478, 2128, 3239, 1962, 2615, 3146, 2761]
watchlist = []
watch_coords = []
watch_frames = [0]
for i in range(1, frame_number // 10 + 1):
    watch_frames.append(i * 10 - 1)
print(watch_frames)

class particle:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight

def sign(x):
    return int(x > 0) * 2 - 1

def cell_to_coord(i):
    return (i-size/2)/(size/scale)

def coord_to_cell(x):
    return round(x*(size/scale) + size/2)

def initial_distribution(x, y):
    return math.atan(y*20 / scale) + math.pi/2

def flow(x, y):
    r = (x*x + y*y)**0.5
    if r == 0:
        return [0.0, 0.0]
    Vt_r = math.tanh(r) / math.cosh(r)**2
    dxdt = speed * (-Vt_r * y / r)
    dydt = speed * (Vt_r * x / r)
    return [dxdt, dydt]

for i in range(size*3):
    for j in range(size*3):
        flow_grid[i][j] = flow(cell_to_coord(j-size), cell_to_coord(i-size))


def get_flow_from_grid(x, y, interpolation):
    if interpolation == 0:
        nearest = [coord_to_cell(y)+size, coord_to_cell(x)+size]  # no interpolation
        dxdt = flow_grid[nearest[0]][nearest[1]][0]
        dydt = flow_grid[nearest[0]][nearest[1]][1]
        # print(dxdt, dydt, 'no inter')
        return dxdt, dydt
    nlt = [0, 0]
    nearest = [cell_to_coord(coord_to_cell(x)), cell_to_coord(coord_to_cell(y))]
    nlt = [coord_to_cell(i) for i in nearest[::-1]]
    if x == nearest[0] and y == nearest[1]:
        dydt = -flow_grid[coord_to_cell(x)+size][coord_to_cell(y)+size][0]
        dxdt = -flow_grid[coord_to_cell(x)+size][coord_to_cell(y)+size][1]
        return dxdt, dydt
    if nearest[0] > x:
        nlt[1] -= 1
    if nearest[1] > y:
        nlt[0] -= 1
    nearest_coords = [[cell_to_coord(nlt[1]), cell_to_coord(nlt[0])],
                      [cell_to_coord(nlt[1]+1), cell_to_coord(nlt[0])],
                      [cell_to_coord(nlt[1]), cell_to_coord(nlt[0]+1)],
                      [cell_to_coord(nlt[1]+1), cell_to_coord(nlt[0]+1)]]
    distances = [((j - x) ** 2 + (i - y) ** 2) ** 0.5 for i, j in nearest_coords]
    nlt[0] += size
    nlt[1] += size
    nearset_weights = [flow_grid[nlt[0]][nlt[1]], flow_grid[nlt[0]][nlt[1]+1],
                       flow_grid[nlt[0]+1][nlt[1]], flow_grid[nlt[0]+1][nlt[1]+1]]
    dxdt, dydt, distsum = 0, 0, 0
    if interpolation == 1:
        for i in range(4):
            dxdt += nearset_weights[i][0] / 4
            dydt += nearset_weights[i][1] / 4
        return dxdt, dydt

    if interpolation == 2:
        for i in range(4):
            distance_coef = 1 / distances[i]
            dxdt += nearset_weights[i][0] * distance_coef
            dydt += nearset_weights[i][1] * distance_coef
            distsum += distance_coef
        dxdt /= distsum
        dydt /= distsum
        return  dxdt, dydt
    raise ValueError('incorrect interpolation parameter')


def particles_to_data(move=True):
    for n in range(len(particles)):
        if move:
            gradient = get_flow_from_grid(particles[n].x, particles[n].y, interpolation=interpolation_type)
            particles[n].x += gradient[0]
            particles[n].y += gradient[1]
            if write_output:
                if n in watchlist:
                    watch_coords[watchlist.index(n)].append([particles[n].x, particles[n].y])
        y, x = particles[n].y, particles[n].x
        i, j = coord_to_cell(y), coord_to_cell(x)
        i_min, i_max = max(i, 0), min(i + spread_rad, size)
        j_min, j_max = max(j, 0), min(j + spread_rad, size)
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

    for i in range(size):
        for j in range(size):
            if distsum_in_pixel[i][j] > 0:
                data[i][j] /= distsum_in_pixel[i][j]
            if distsum_in_pixel[i][j] < generate_threshold[0] or \
                    particles_in_pixel[i][j] < generate_threshold[1]:
                particles.append(particle(cell_to_coord(j), cell_to_coord(i), data[i][j]))

def just_move():
    for n in range(len(particles)):
        gradient = get_flow_from_grid(particles[n].x, particles[n].y, interpolation=interpolation_type)
        # gradient = flow(particles[n].x, particles[n].y)
        particles[n].x += gradient[0]
        particles[n].y += gradient[1]


begin_time = time.time()

for i in range(-size//2, size*3//2):    ##############  НАЧАЛО КОДА  ###################
    for j in range(-size//2, size*3//2):
        if i % particle_density != 0 or j % particle_density != 0:
            continue
        # if j != size/2:
        #     continue
        x = cell_to_coord(j)
        y = cell_to_coord(i)
        # if len(particles)-1 in watchlist:
        watchlist.append(len(watchlist))
        weight = 1
        watch_coords.append([[x, y]])
        # else:
        #     weight = 1
        # weight = initial_distribution(x, y, scale)
        particles.append(particle(x + scale / size, y + scale / size, weight))


filenames.append('out_images\\plot0.png')
particles_to_data(False)
data_np = numpy.array(data)
plt.imsave(filenames[-1], data_np, cmap=pallete)

end_time = time.time()
print(end_time - begin_time)
total_time = 0

run_till = frame_number
for k in range(1, frame_number):
    while run_till <= k:
        run_till = int(input("New run_till:"))
    if (k+1) % 10 == 0:
        print(k, 'time elapsed =', total_time)
    begin_time = time.time()
    data = [[0 for i in range(size)] for j in range(size)]
    distsum_in_pixel = [[0 for i in range(size)] for j in range(size)]
    particles_in_pixel = [[0 for i in range(size)] for j in range(size)]

    filenames.append('out_images\\plot' + str(k) + '.png')
    particles_to_data()
    data_np = numpy.array(data)
    plt.imsave(filenames[-1], data_np, cmap=pallete)

    end_time = time.time()
    total_time += end_time - begin_time

if write_output:
    f = open('out_stats\\third_problem_watch_inter_' + str(interpolation_type) + '.txt', "w")
    f.write('size = ' + str(size) + ', frame_number = ' + str(frame_number) +
            ', scale = ' + str(scale) + ', particle_density = ' + str(particle_density) +
            ', speed = ' + str(speed) + ', interpolation_type = ' + str(interpolation_type) + '\n')
    f.write('watchlist: ')
    for i in watchlist:
        f.write(str(i))
        f.write(' ')
    f.write('\n')
    f.write('watch_frames: ')
    for i in watch_frames:
        f.write(str(i))
        f.write(' ')
    f.write('\n')
    for particle_number in range(len(watchlist)):
        for frame in watch_frames:
            f.write(str(watch_coords[particle_number][frame][0]))
            f.write(' ')
            f.write(str(watch_coords[particle_number][frame][1]))
            f.write(' ')
        f.write('\n')
    f.close()

with imageio.get_writer('out_mov\mov.gif',
                        mode='I', duration=0.017) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

particles_to_data()
data_np = numpy.array(data)
plt.imsave('out_images\\plot' + '_last' + '.png', data_np, cmap=pallete)

plt.figure(figsize=(5,5))
plt.imshow(data_np, cmap=pallete)
plt.show()

total_time += end_time - begin_time
print(total_time)

average_distanse_from_center = []
for i in range(len(watch_frames)):
    sum_ = 0
    for j in range(len(watchlist)):
        sum_ += ((watch_coords[j][i][0]**2 + watch_coords[j][i][1]**2))**0.5
    average_distanse_from_center.append(sum_ / len(watchlist))
print(average_distanse_from_center)


