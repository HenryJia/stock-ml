import csv #, numpy

def get_last(filename, n):

    file = open(filename)
    reader = csv.reader(file)

    complete_array = []

    for row in reader: # Create an array of the file
        complete_array.append(row)

    output_array = complete_array[-n:] # Output = last n rows

    output_array_float = []

    for i in output_array:
        line = []
        for j in i:
            line.append(float(j))
        output_array_float.append(line)

    #return numpy.matrix(output_array) # Convert into matrix

    return output_array_float

#print(concat('ARM_LD.txt', 3))  #TEST
    
