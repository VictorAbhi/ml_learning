import numpy as np
 
def add_matrices(matrix1, matrix2):
    return matrix1 + matrix2
 
def subtract_matrices(matrix1, matrix2):
    return matrix1 - matrix2
 
def multiply_matrices(matrix1, matrix2):
    return np.dot(matrix1, matrix2)
 
def transpose_matrix(matrix):
    return matrix.T
 
def main():
    print("Simple Matrix Calculator")
    print("Enter the dimensions of your matrices (rows and columns):")
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))
 
    print("Enter the elements of the first matrix row by col (space-separated):")
    matrix1 = np.array([list(map(float, input().split())) for _ in range(rows)])

    print(matrix1)
    print("Enter the elements of the second matrix row by row (space-separated):")
    matrix2 = np.array([list(map(float, input().split())) for _ in range(rows)])
    
 
    while True:
        print("\nChoose an operation:")
        print("1. Add Matrices")
        print("2. Subtract Matrices")
        print("3. Multiply Matrices")
        print("4. Transpose First Matrix")
        print("5. Transpose Second Matrix")
        print("6. Exit")
        choice = int(input("Enter your choice (1-6): "))
        
        if choice == 1:
            result = add_matrices(matrix1, matrix2)
            print("Result of Addition:\n", result)
        elif choice == 2:
            result = subtract_matrices(matrix1, matrix2)
            print("Result of Subtraction:\n", result)
        elif choice == 3:
            if matrix1.shape[1] == matrix2.shape[0]:
                result = multiply_matrices(matrix1, matrix2)
                print("Result of Multiplication:\n", result)
            else:
                print("Matrix multiplication is not possible. Ensure the number of columns in the first matrix equals the number of rows in the second matrix.")
        elif choice == 4:
            result = transpose_matrix(matrix1)
            print("Transpose of First Matrix:\n", result)
        elif choice == 5:
            result = transpose_matrix(matrix2)
            print("Transpose of Second Matrix:\n", result)
        elif choice == 6:
            print("Exiting the Matrix Calculator. Goodbye!")
            break
        else:
            print("Invalid choice! Please try again.")
 
if __name__ == "__main__":
    main()