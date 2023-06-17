import math

import numpy

class particle:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight

class patch:
    def __init__(self, x_begin, y_begin, x_end=None, y_end=None, value_begin=None, value_end=None):
        self.x_begin = x_begin
        self.y_begin = y_begin
        self.x_end = x_end
        self.y_end = y_end
        self.value_begin = value_begin
        self.value_end = value_end

def cell_to_coord(i, size, scale):
    return (i-size/2)/(size/scale)

def coord_to_cell(x, size, scale):
    return round(x*(size/scale) + size/2)

def initial_distribution(x, y, scale):
    # if x == 1 and y == -1:
    #     return 1
    # return 0
    return math.atan(y*20 / scale) + math.pi/2

def flow(x, y, this_one=False):
    r = (x*x + y*y)**0.5
    if r == 0:
        return [0.0, 0.0]
    Vt_r = math.tanh(r) / math.cosh(r)**2
    dxdt = 0.05 * (-Vt_r * y / r)
    dydt = 0.05 * (Vt_r * x / r)
    # if this_one:
    #     print('dxdt, dydt =', dxdt, dydt)
    # print('dxdt, dydt', dxdt, dydt)
    # dxdt = 0.005*y * normal_distribution(math.sqrt(x*x + y*y), 0.1, 0.25)
    # dydt = -0.005*x * normal_distribution(math.sqrt(x*x + y*y), 0.1, 0.25)
    return [dxdt, dydt]

def move_particles(particles, particles_in_pixel, size, scale, data):
    for n in range(len(particles)):
        gradient = flow(particles[n].x, particles[n].y)  #предварительно сдвинув по течению
        particles[n].x += gradient[0]
        particles[n].y += gradient[1]
        i = coord_to_cell(particles[n].y, size, scale)
        j = coord_to_cell(particles[n].x, size, scale)
        # if i >= size or i < 0 or j >= size or j < 0:
        #     continue
        # data[i][j] += particles[n].weight
        # particles_in_pixel[i][j].append(n)
        for i1 in range(max(i-3, 0), min(i+3, size-1)):
            for j1 in range(max(j-3, 0), min(j+3, size-1)):
                data[i1][j1] += particles[n].weight  # * ((i1 - i)*(i1 - i) + (j1 - j)*(j1 - j))**0.5
                particles_in_pixel[i1][j1].append(n)

def merge_particles(particles, particles_in_pixel, particles_new, size):
    for i in range(size):               # переписываем частицы в новый массив,
        for j in range(size):           # если больше одной в пискеле - сливаем в одну
            ps_temp = [particles[i] for i in particles_in_pixel[i][j]]
            # if i == 4 and j == 7:
            #     print([i.weight for i in ps_temp])
            length = len(ps_temp)
            if length == 1:
                particles_new.append(ps_temp[0])
                # data[i][j] = ps_temp[0].weight
            elif length > 1:
                avg_x = sum([ps_temp[k].x for k in range(length)]) / length
                avg_y = sum([ps_temp[k].y for k in range(length)]) / length
                avg_weight = sum([ps_temp[k].weight * ((ps_temp[k].x - avg_x)**2
                        + (ps_temp[k].y - avg_x)**2)**0.5 for k in range(length)]) / length
                # avg_weight = sum([ps_temp[k][2] for k in range(length)]) / length
                particles_new.append(particle(avg_x, avg_y, avg_weight))
                # data[i][j] = avg_weight

def fix_edges(particles_in_pixel, particles_new, size, scale):
    for i in range(size-1):   # обходим границы
        if len(particles_in_pixel[0][i]) == 0:
            x, y = cell_to_coord(i, size, scale), cell_to_coord(0, size, scale)
            particles_new.append(particle(x, y, initial_distribution(x, y, scale)))
            particles_in_pixel[0][i].append(1)
        if len(particles_in_pixel[size-1][i+1]) == 0:
            x, y = cell_to_coord(i+1, size, scale), cell_to_coord(size-1, size, scale)
            particles_new.append(particle(x, y, initial_distribution(x, y, scale)))
            particles_in_pixel[size-1][i+1].append(1)
        if len(particles_in_pixel[i+1][0]) == 0:
            x, y = cell_to_coord(0, size, scale), cell_to_coord(i+1, size, scale)
            particles_new.append(particle(x, y, initial_distribution(x, y, scale)))
            particles_in_pixel[i+1][0].append(1)
        if len(particles_in_pixel[i][size-1]) == 0:
            x, y = cell_to_coord(size-1, size, scale), cell_to_coord(i, size, scale)
            particles_new.append(particle(x, y, initial_distribution(x, y, scale)))
            particles_in_pixel[i][size-1].append(1)

def patch_holes(particles_in_pixel, particles_to_add, data, size):
    for i in range(size):
        patch_x = patch(0, i, value_begin=data[i][0])
        for j in range(1, size):
            if len(particles_in_pixel[i][j]) != 0:
                if patch_x.x_begin != j - 1:
                    patch_x.x_end = j
                    patch_x.value_end = data[i][j] / 2
                    length = patch_x.x_end - patch_x.x_begin - 1
                    for k in range(patch_x.x_begin + 1, patch_x.x_end):
                        particles_to_add[i][k] = patch_x.value_begin * (length - k - 1) / length + \
                                                 patch_x.value_end * (k + 1) / length
                patch_x.x_begin = j
                patch_x.value_begin = data[i][j]
    for j in range(size):
        patch_y = patch(j, 0, value_begin=data[0][j])
        for i in range(1, size):
            if len(particles_in_pixel[i][j]) != 0:
                if patch_y.y_begin != i - 1:
                    patch_y.y_end = i
                    patch_y.value_end = data[i][j] / 2
                    length = patch_y.y_end - patch_y.y_begin - 1
                    for k in range(patch_y.y_begin + 1, patch_y.y_end):
                        particles_to_add[k][j] += patch_y.value_begin * (length - k - 1) / length + \
                                                 patch_y.value_end * (k + 1) / length
                patch_y.y_begin = i
                patch_y.value_begin = data[i][j]

def add_particles_to_data(particles_new, particles_to_add, data, old_length, size, scale):
    for i in range(size):
        for j in range(size):
            if particles_to_add[i][j] != None:
                particles_new.append(particle(cell_to_coord(j, size, scale), cell_to_coord(i,
                                                        size, scale), particles_to_add[i][j]))
    for i in range(old_length, len(particles_new)):
        data[coord_to_cell(particles_new[i].y, size, scale)][coord_to_cell(particles_new[i].x,
                                                    size, scale)] = particles_new[i].weight