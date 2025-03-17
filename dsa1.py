# # def linear_search(arr, target):
# #     """
# #     Perform a linear search for the target in the given array.

# #     Parameters:
# #     arr (list): The list to search through.
# #     target: The element to search for.

# #     Returns:
# #     int: The index of the target if found, otherwise -1.
# #     """
# #     # Iterate over each element in the array
# #     for index, element in enumerate(arr):
# #         # Check if the current element is the target
# #         if element == target:
# #             # If found, return the index
# #             return index
# #     # If the target is not found, return -1
# #     return -1

# # # Example usage
# # if __name__ == "__main__":
# #     numbers = [10, 20, 30, 40, 50]
# #     target_number = 30
# #     result = linear_search(numbers, target_number)
    
# #     if result != -1:
# #         print(f"Element found at index {result}")
# #     else:
# #         print("Element not found")




# # # x=10
# # # y=5


# # # print(x+y)
# # # print(x-y)
# # # print(x*y)
# # # print(x/y)
# # # print(x%y)
# # # print(x**y)
# # # print(x//y)
# # # print(x==y)



# # #comparision operator 



# # # print(x>y)
# # # print(x<y)
# # # print(x!=y)
# # # print(x>=y)
# # # print(x<=y)



# # #logical operator

# # # this is membership operator 



# # fruits=['apple','banana','cherry']


# # print('apple' in fruits)

# # print('banana' in fruits )

# # print('orage' in fruits)



# # # data type in python 

# # # how to check the data type in python



# # x=5

# # print(type(x))



# # x=5.5


# # print(type(x))


# # x=True

# # print(type(x))


# # z=2+3j



# # print(type(z))



# # # mutable objects can be changed in place without creating a new object

# # # Immmutable objects cannot be changed  



# # # strings ::


# # s='Hello'

# # print(s[0])  # it will acess the first element of the string 

# # print(s.upper())  # it will convert the string into upper case 





# # # tupeles 

# # coordinates=(10,20,30,40)

# # print(coordinates[1])  # acess second element of the tuples


# # print(len(coordinates))  # it will give length of the tuple s


# # # tuples are immutable 


# # # coordinates[1]=25  # it will give errr because tuples are immutable




# # #mutable data types in python


# # #lists 

# # fruits=['apple','banana','cherry']

# # fruits[1]='orange'

# # fruits.append('date')

# # print(fruits)

# # # list can grows shrink or have their elements modified

# # lets lookk at ways to reference elements list items. The following progrm print out



# name=['Viivi','Ahmed','Pekka','Olga','Mary','Pekka']


# print(name[3])

# print(name[1])

# print(name[-2])


# print(name[1:3])


# print(name[2:])


# print(name)


# name.remove('Pekka')  # it will remove the element from the list 

# print(name)



# #


# dictionaries in python 



# person={
#     'name':'John',
#     'age':30,
#     'city':'New York'

# }



# print(person)


# print(type(person))


# dictionaries store key value pairs and allow modification of the values stored in the dictionary  











#sets in python




# unique_numbers={1,2,3,4}


# unique_numbers.add(5)


# print(unique_numbers)
# print(type(unique_numbers))




# Control Statements in python 


# if else statement


# if-else

# x=10

# if x>5:
#     print('x is geater than 5')


# elif x==5:
#     print('x is equal to 5')


# else:
#     print('x is less than 5')


# for loop : The for loop is used for iterating over a sequence (a list,tuple, or range) or  other  iterable objects 



# while loo : The while loop is used to repeat as long as a condition is Ture 


# The loop terminates  when  the specified condition is False


# for loop 
# # for i in range(1,6):
#         print(i)



# for i in range(1,6):
#     print(i)


# while loop 

# count=0


# while count<3:
#     print(count)
#     count+=1



# rounds=int(input("How many greetings"))


# finished_rounds=0
# while finished_rounds<rounds:
#     print("good morning")
   

#     finished_rounds=finished_rounds+1



# f8unctions :



# Using functions helps to to avoid situations where you would have to write the same code multiple time




# function call
# using functions helps you avoid situations where you would have to write and copy the same  or a simmilar block of code to various part of your programms 
# Resuing the same code should always be avoided in programming as it makes program more complex

# function defination ::





# def calculate_rectangle_area(length,width):

#     """" this function calculates the area of a rectangle given its length and width """

#     area=length*width
#     return area


# length=float(input("enter the length of the rectange:"))


# width=float(input("enter the width of the rectangle:"))


# area=calculate_rectangle_area(length,width)



# print(f"The area of the rectangle is {area}")




# Input and formatting in python

#taking input::




# name=input("enter your name::")



# print(f"Hello my name is {name}")



# exception handling in python



# the try and except blocks in python are usedd to handle for excception handling. They allow to catch and handle errors 

# try block 


# The try block is used to wrap code that might raise an exception


# if no exception occurs the code in the try block is excuted normally 
# if an exception occurs the  rest code in the try block is skipped and python jumps to the except block 

#finally block 



# try-except block 

# try:
#     number=float(input("enter a number"))


#     print(10/number)

# except ZeroDivisionError:
#     print("cannot divide by zero")













 











