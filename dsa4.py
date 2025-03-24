# Implementation of Binary Search Algorithm in Python

def binary_search(arr, target):
    """
    Perform binary search on a sorted array to find the target element.

    :param arr: List of sorted elements
    :param target: Element to search for
    :return: Index of the target element if found, else -1
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        # Check if the target is at mid
        if arr[mid] == target:
            return mid
        # If target is smaller, ignore the right half
        elif arr[mid] > target:
            right = mid - 1
        # If target is larger, ignore the left half
        else:
            left = mid + 1

    # Target not found
    return -1

# Example usage
if __name__ == "__main__":
    array = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7

    result = binary_search(array, target)
    if result != -1:
        print(f"Element {target} found at index {result}.")
    else:
        print(f"Element {target} not found in the array.")