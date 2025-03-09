# Finally block is used to exceute code after try-except block regardless of the result of the try-except block 




try:
    file=open('example.txt','r')

    print(file)

except FileNotFoundError:
    print("File not found")


finally:
    print("Finally block always excuted ")