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


