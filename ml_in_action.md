create the dataset 

```py
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = np.array(['A', 'A', 'B', 'B'])  # Convert to NumPy array for boolean indexing
    return group, labels
```

group, labels = createDataSet()

how to re;ate groups and labels 

```py

# Create boolean condition
condition = labels == 'A'
print("Condition:", condition)

# Apply boolean indexing to filter group
group_A = group[condition]
print("Group A:", group_A)

```
