import sys
import numpy as np

def open_data(file, directionsOnly):
    data_file = open(file, 'r')
    data = []
    for line in data_file:
        data_array = line.strip().split("|")
        data_array = [int(x) for x in data_array]
        if directionsOnly:
            if data_array[1] != 0:
                data.append(data_array)
        else:
            data.append(data_array)
    return np.array(data)

def write_to_file(file, data):
	write_file = open(file, 'w')
	data = [[str(data[i, j]) for j in range(data.shape[1])] for i in range(data.shape[0])]
	for data_row in data:
		write_file.write("|".join(data_row)+"\n")

def add_1_to_inputs(data):
	data[:, 0:2] = np.add(data[:, 0:2], np.array([[1]]))
	return data

def move_inputs_up(data):
	for x in range(data.shape[0]-1):
		data[x, 0:2] = data[x+1, 0:2]
	return data[:data.shape[0]-1]

def flip_x(x):
	if x == -1:
		return -1
	max_x = 600
	return max_x-x

def flip_y(y):
	if y == -1:
		return -1
	max_y = 530
	return max_y-y

def flip_direction_horizontal(direction):
	if direction == 2:
		return 4
	elif direction == 4:
		return 2
	else:
		return direction

def flip_direction_vertical(direction):
	if direction == 1:
		return 3
	elif direction == 3:
		return 1
	else:
		return direction

def flip_horizontal(data):
	data = np.copy(data)
	for r in range(len(data)):
		data[r][0] = flip_direction_horizontal(data[r][0])
		data[r][1] = flip_direction_horizontal(data[r][1])
		data[r][2] = flip_x(data[r][2]) #player 1
		data[r][4] = flip_direction_horizontal(data[r][4])
		data[r][7] = flip_x(data[r][7]) #player 2
		data[r][9] = flip_direction_horizontal(data[r][9])
		data[r][12] = flip_x(data[r][12]) #enemy 1
		data[r][14] = flip_direction_horizontal(data[r][14])
		data[r][16] = flip_x(data[r][16]) #enemy 2
		data[r][18] = flip_direction_horizontal(data[r][18])
		data[r][20] = flip_x(data[r][20]) #enemy 3
		data[r][22] = flip_direction_horizontal(data[r][22])
		data[r][24] = flip_x(data[r][24]) #enemy 4
		data[r][26] = flip_direction_horizontal(data[r][26])
		data[r][28] = flip_x(data[r][28]) #snowball1
		data[r][30] = flip_direction_horizontal(data[r][30])
		data[r][31] = flip_x(data[r][31]) #snowball2
		data[r][33] = flip_direction_horizontal(data[r][33])
		data[r][34] = flip_x(data[r][34]) #snowball3
		data[r][36] = flip_direction_horizontal(data[r][36])
		data[r][37] = flip_x(data[r][37]) #snowball4
		data[r][39] = flip_direction_horizontal(data[r][39])
		data[r][40] = flip_x(data[r][40]) #snowball5
		data[r][42] = flip_direction_horizontal(data[r][42])
		data[r][43] = flip_x(data[r][43]) #snowball6
		data[r][45] = flip_direction_horizontal(data[r][45])
		data[r][46] = flip_x(data[r][46]) #snowball7
		data[r][48] = flip_direction_horizontal(data[r][48])
		data[r][49] = flip_x(data[r][49]) #snowball8
		data[r][51] = flip_direction_horizontal(data[r][51])
	return data

def flip_vertical(data):
	data = np.copy(data)
	for r in range(len(data)):
		data[r][0] = flip_direction_vertical(data[r][0])
		data[r][1] = flip_direction_vertical(data[r][1])
		data[r][3] = flip_y(data[r][3]) #player 1
		data[r][4] = flip_direction_vertical(data[r][4])
		data[r][8] = flip_y(data[r][8]) #player 2
		data[r][9] = flip_direction_vertical(data[r][9])
		data[r][13] = flip_y(data[r][13]) #enemy 1
		data[r][14] = flip_direction_vertical(data[r][14])
		data[r][17] = flip_y(data[r][17]) #enemy 2
		data[r][18] = flip_direction_vertical(data[r][18])
		data[r][21] = flip_y(data[r][21]) #enemy 3
		data[r][22] = flip_direction_vertical(data[r][22])
		data[r][25] = flip_y(data[r][25]) #enemy 4
		data[r][26] = flip_direction_vertical(data[r][26])
		data[r][28] = flip_y(data[r][29]) #snowball1
		data[r][30] = flip_direction_vertical(data[r][30])
		data[r][32] = flip_y(data[r][32]) #snowball2
		data[r][33] = flip_direction_vertical(data[r][33])
		data[r][35] = flip_y(data[r][35]) #snowball3
		data[r][36] = flip_direction_vertical(data[r][36])
		data[r][38] = flip_y(data[r][38]) #snowball4
		data[r][39] = flip_direction_vertical(data[r][39])
		data[r][41] = flip_y(data[r][41]) #snowball5
		data[r][42] = flip_direction_vertical(data[r][42])
		data[r][44] = flip_y(data[r][44]) #snowball6
		data[r][45] = flip_direction_vertical(data[r][45])
		data[r][47] = flip_y(data[r][47]) #snowball7
		data[r][48] = flip_direction_vertical(data[r][48])
		data[r][50] = flip_y(data[r][50]) #snowball8
		data[r][51] = flip_direction_vertical(data[r][51])
	return data

def main():
	filename = sys.argv[1]
	shouldAugment = False
	directionsOnly = False
	for x in range(2, len(sys.argv)):
		if sys.argv[x] == "-a":
			shouldAugment = True
		if sys.argv[x] == "-d":
			directionsOnly = True
	data = open_data(filename, directionsOnly)
	data = add_1_to_inputs(data)
	data = move_inputs_up(data)
	newfilename = filename[:len(filename)-4]+"Parsed"
	if shouldAugment:
		data_flip_h = flip_horizontal(data)
		data_flip_v = flip_vertical(data)
		data_flip_hv = flip_vertical(data_flip_h)
		data = np.concatenate((data, data_flip_h))
		data = np.concatenate((data, data_flip_v))
		data = np.concatenate((data, data_flip_hv))
		newfilename += "Augmented"
	if directionsOnly:
		newfilename += "DirectionsOnly"
	newfilename += ".txt"
	write_to_file(newfilename, data)

main()