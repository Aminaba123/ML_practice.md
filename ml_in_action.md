create the dataset 

```py
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])  # Convert to NumPy array for boolean indexing
    return group, labels
```

```py
group, labels = createDataSet()
```


how to re;ate groups and labels 

```py

# Create boolean condition
condition = labels == 'A'
print("Condition:", condition)

# Apply boolean indexing to filter group
group_A = group[condition]
print("Group A:", group_A)

```
how `group[labels == 'A']` works?

The expression `group[labels == 'A']` is an example of boolean indexing in NumPy

Boolean Indexing Concept:

Boolean indexing is a method of selecting data from an array based on a condition. It involves creating a boolean array where each element is True or False depending on whether the condition is met.

Components of the Expression:

`labels == 'A'`: This is a boolean condition applied to the labels array.
`group[...]`: This is the array you want to filter based on the boolean condition.


```py

group_A = group[np.array(labels) == 'A']
group_B = group[np.array(labels) == 'B']

```

what does this mean ?

```py
group_A = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])

# Select all rows and the first column
first_column = group_A[:, 0]

```

`group_A[:, 0]`:

`:` means `select all rows.`
`0` means `select the first column.`

plot the points 

```py

import numpy as np 
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels 

group, labels = createDataSet()
print(group, labels)

# Separate the data points by labels

print(np.array(labels))
group_A = group[np.array(labels) == 'A']
group_B = group[np.array(labels) == 'B']

# Plot the data points

plt.scatter(group_A[:, 0], group_A[:, 1], color='red', label='Class A')
plt.scatter(group_B[:, 0], group_B[:, 1], color='blue', label='Class B')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Points')
plt.legend()

# Show the plot
plt.show()

```

#

here we can see how to read ad parse the txt fil

```py

import os
import numpy as np

def file2matrix(filename):
    """
    Checks if the file exists, reads data from the file, and converts it into a NumPy matrix and label vector.

    Parameters:
    filename (str): The name of the file to process.

    Returns:
    tuple: A tuple containing:
        - np.ndarray: Matrix containing the features.
        - list: List containing the labels.
    """
    # Check if the file exists in the current directory
    if filename in os.listdir():
        print(f"File '{filename}' found in the current directory.")
        
        lines = []
        try:
            # Open and read the file
            with open(filename, 'r') as txtfile:
                for line in txtfile:
                    line = line.strip()  # Strip leading and trailing whitespace
                    listFromLine = line.split('\t')  # Split the line by tabs
                    lines.append(listFromLine)  # Append the line to the list
        
        except FileNotFoundError:
            print('There is no such file in this directory')
            return None, None  # Return None if file not found

        # Convert collected data to NumPy matrix and label vector
        numberOfLines = len(lines)  # Get the number of lines
        returnMat = np.zeros((numberOfLines, 3))  # Initialize a matrix to hold the data
        classLabelVector = []  # Initialize a list to hold the labels

        for index, line in enumerate(lines):
            returnMat[index, :] = line[0:3]  # Assign the first three elements to the matrix
            classLabelVector.append(int(line[-1]))  # Assign the last element as the label
        
        return returnMat, classLabelVector  # Return the matrix and label vector
    
    else:
        print(f"File '{filename}' does not exist in the current directory.")
        return None, None  # Return None if file does not exist

# Example usage
filename = 'datingTestSet2.txt'
returnMat, classLabelVector = file2matrix(filename)

if returnMat is not None and classLabelVector is not None:
    print("Matrix:")
    print(returnMat)
    print("Labels:")
    print(classLabelVector)
else:
    print("No data to process.")


```

#

Some useful code for reading data 







```py

def csv_reader(fname):
    with open(fname, 'r') as f:
        out = list(csv.reader(f))
    return out


def get_files(folds_path: str, fold: int, split: str) -> list:
    csv_dir = os.path.join(folds_path, 'fold_{}'.format(fold))
    csv_file = os.path.join(csv_dir, '{}_f_{}.csv'.format(split, fold))
    if os.path.exists(csv_file):
        files = csv_reader(csv_file)
    else:
        raise FileExistsError('File {} not found.'.format(csv_file))
    return files


def decode_classes(files: list, classes: dict) -> list:
    files_decoded_classes = []
    for f in files:
        class_name = f[0].split('/')[2]
        files_decoded_classes.append((f[0], classes[class_name]))

    return files_decoded_classes

```

#
