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
speed = 0.01
write_output = True
pallete = 'gist_heat'


data = [[0 for i in range(size)] for j in range(size)]    #пиксели, отрисовка
particles = []  #одномерный массив частиц, которые плавают
distsum_in_pixel = [[0 for i in range(size)] for j in range(size)] # сумма расстояний
# влияющих пискелей, аналог particles_in_pixel
particles_in_pixel = [[0 for i in range(size)] for j in range(size)]
filenames = []
watchlist = [4690, 3152, 1787, 3478, 2128, 3239, 1962, 2615, 3146, 2761]
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

def rk4_step(x, y, h):
    k1 = flow(x, y)
    k2 = flow(x + h/2 * k1[0], y + h/2 * k1[1])
    k3 = flow(x + h/2 * k2[0], y + h/2 * k2[1])
    k4 = flow(x + h * k3[0], y + h * k3[1])
    dxdt = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6
    dydt = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6
    return [dxdt, dydt]

def particles_to_data(move=True):
    for n in range(len(particles)):
        if move:
            for i in range(5):
                gradient = rk4_step(particles[n].x, particles[n].y, speed)
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
        gradient = rk4_step(particles[n].x, particles[n].y, speed)
        particles[n].x += gradient[0]
        particles[n].y += gradient[1]


begin_time = time.time()

# x = -0.25
# y = -0.25
# weight = 1
# watchlist.append(len(particles))
# watch_coords.append([[x, y]])
# particles.append(particle(x + scale / size, y + scale / size, weight))
# weight = 2
# watchlist.append(len(particles))
# watch_coords.append([[x, y]])
# particles.append(particle(x + scale / size, y + scale / size, weight))

for i in range(-size//2, size*3//2):    ##############  НАЧАЛО КОДА  ###################
    for j in range(-size//2, size*3//2):
        if i % particle_density != 0 or j % particle_density != 0:
            continue
        # if j != size/2:
        #     continue
        x = cell_to_coord(j)
        y = cell_to_coord(i)
        if len(particles)-1 in watchlist:
            weight = 2
            watch_coords.append([[x, y]])
        else:
            weight = 1
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
    f = open('out_stats\\runge_kutta_watch.txt', "w")
    f.write('size = ' + str(size) + ', frame_number = ' + str(frame_number) +
            ', scale = ' + str(scale) + ', particle_density = ' + str(particle_density) +
            ', speed = ' + str(speed) + '\n')
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


