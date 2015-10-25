import csv, os

#os.chdir(r'C:\Users\Alfie\Documents\Perceptron\Stuff')

def reformat(filename): # The main function
    
    print(filename)
    
    original_file = open(filename)
    reader = csv.reader(original_file)

    # Create files to write to in subfolders data and targets
    main_file = os.path.join(filename[:-4] + 'X' + filename[-4:]) # This should insert an X into the filename before the file extension (assuming the extension 

    main_data = open(main_file, 'w')

    data_array = []
    line_to_write = ''

    for row in reader: # Create an array of all the cells in the original file
        data_array.append(row)

    for row in range(1,len(data_array)): # Cycle through all
        for column in range(4,9): # For columns 5 to 9 inclusive
            line_to_write += str(data_array[row][column] + ',') # Add the value of this cell to the (temporary) line to be written

        #print(line_to_write)
        main_data.write(line_to_write[:-1] + '\n') # Wirte the 'line to write' to the main data file, but without the final comma ([:-1]) and with a line break ('\n')
        line_to_write = ''


#list_of_files = open('symbols_processed3.csv')
#list_reader = csv.reader(list_of_files)

#for row in list_reader: # Perform the reformat function using the name of each file in the list as the filename
    #for col in row:
        #reformat(str(col))
