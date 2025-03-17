def gini_index(groups, classes):
    # Count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    
    # Sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # Avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # Score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # Weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Example usage
groups = [
    [[1, 1], [2, 1], [3, 0]],
    [[4, 0], [5, 0]]
]
classes = [0, 1]
gini = gini_index(groups, classes)
print(f'Gini Index: {gini}')




# the true positive rate (TPR) and false positive rate (FPR) are defined as follows:


# how can we calculate the 
