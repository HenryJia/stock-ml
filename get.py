import csv #, numpy

def get(filename, m, n):

    file = open(filename)
    reader = csv.reader(file)

    complete_array = []

    for row in reader: # Create an array of the file
        complete_array.append(row)

    if (m > 0 and n > 0):
      output_array = complete_array[-m:-n] # Output = last m to n rows
    elif (m < 0 and n > 0):
      output_array = complete_array[:-n] # Output = last m to n rows
    elif (m > 0 and n < 0):
      output_array = complete_array[-m:] # Output = last m to n rows
    else:
      output_array = complete_array

    output_array_float = []

    for i in output_array:
        line = []
        for j in i:
            line.append(float(j))
        output_array_float.append(line)

    #return numpy.matrix(output_array) # Convert into matrix

    return output_array_float
  
def get_all(filename):

    file = open(filename)
    reader = csv.reader(file)

    output_array = []

    for row in reader: # Create an array of the file
        output_array.append(row)

    output_array_float = []

    for i in output_array:
        line = []
        for j in i:
            line.append(float(j))
        output_array_float.append(line)

    #return np.array(output_array) # Convert into matrix

    return output_array_float

#print(concat('ARM_LD.txt', 3))  #TEST
    
