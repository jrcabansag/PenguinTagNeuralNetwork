import sys
import numpy as np

def open_data(file):
    data_file = open(file, 'r')
    data = []
    for line in data_file:
        data_array = line.strip().split("|")
        data.append(data_array)
    return np.array(data)

def write_to_file(file, data):
	write_file = open(file, 'w')
	for data_row in data:
		write_file.write("|".join(data_row)+"\n")

def move_inputs_up(data):
	for x in range(data.shape[0]-1):
		data[x, 0:2] = data[x+1, 0:2]
	return data[:data.shape[0]-1]

def main():
	filename = sys.argv[1]
	data = open_data(filename)
	data = move_inputs_up(data) 
	newfilename = filename[:len(filename)-4]+"Parsed.txt"
	write_to_file(newfilename, data)

main()