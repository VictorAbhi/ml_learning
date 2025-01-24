import numpy as np
 
def celsius_to_other(celsius_temps):
    fahrenheit = celsius_temps * 9/5 + 32 
    kelvin = celsius_temps + 273.15
    return fahrenheit, kelvin
 
def fahrenheit_to_other(fahrenheit_temps):
    celsius = (fahrenheit_temps - 32) * 5/9
    kelvin = celsius + 273.15
    return celsius, kelvin
 
def kelvin_to_other(kelvin_temps):
    celsius = kelvin_temps - 273.15
    fahrenheit = celsius * 9/5 + 32
    return celsius, fahrenheit # return the converted temperatures
 
def main():
    print("Temperature Converter with NumPy")
    print("Choose the scale of your input temperatures:")
    print("1. Celsius")
    print("2. Fahrenheit")
    print("3. Kelvin")
    choice = int(input("Enter your choice (1/2/3): "))
    
    if choice not in [1, 2, 3]:
        print("Invalid choice! Please restart the program.") # if choice not in [1, 2, 3], print this message
        return
    
    input_temps = input("Enter a list of temperatures separated by spaces: ")
    temp_array = np.array(list(map(float, input_temps.split()))) # convert input_temps to a list of floats and then to a NumPy array
    
    if choice == 1:
        fahrenheit, kelvin = celsius_to_other(temp_array)
        print("Input Temperatures (Celsius):", temp_array)
        print("Converted to Fahrenheit:", fahrenheit)
        print("Converted to Kelvin:", kelvin)
    elif choice == 2:
        celsius, kelvin = fahrenheit_to_other(temp_array)
        print("Input Temperatures (Fahrenheit):", temp_array)
        print("Converted to Celsius:", celsius)
        print("Converted to Kelvin:", kelvin)
    elif choice == 3:
        celsius, fahrenheit = kelvin_to_other(temp_array)
        print("Input Temperatures (Kelvin):", temp_array)
        print("Converted to Celsius:", celsius)
        print("Converted to Fahrenheit:", fahrenheit)
 
if __name__ == "__main__":
    main()