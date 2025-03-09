def linear_search(arr, target):
    """
    Perform a linear search for the target in the given array.

    Parameters:
    arr (list): The list to search through.
    target: The element to search for.

    Returns:
    int: The index of the target if found, otherwise -1.
    """
    # Iterate over each element in the array
    for index, element in enumerate(arr):
        # Check if the current element is the target
        if element == target:
            # If found, return the index
            return index
    # If the target is not found, return -1
    return -1

# Example usage
if __name__ == "__main__":
    numbers = [10, 20, 30, 40, 50]
    target_number = 30
    result = linear_search(numbers, target_number)
    
    if result != -1:
        print(f"Element found at index {result}")
    else:
        print("Element not found")