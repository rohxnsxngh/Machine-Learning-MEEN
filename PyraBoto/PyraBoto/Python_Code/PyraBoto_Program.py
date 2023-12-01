#!/usr/bin/env python
import serial
import time
import matplotlib.pyplot as plt
import numpy as np
import serial.tools.list_ports
from math import exp
import matplotlib.image as mpimg
from numpy import log as ln
import xlsxwriter


######################### MAIN PROGRAM #########################

def main():
    
    ######################### DISPLAY INSTRUCTIONS #########################
    
    def display_instructions(path):
        
        # Define the instuction image
        instructions = mpimg.imread(path)
        
        # Add the instructions to a plot
        plt.imshow(instructions)
        
        # Hide the axes of the plot
        plt.axis('off')
        
        # Improve figure quality        
        plt.rcParams['figure.dpi'] = 500
        
        # Show the plot
        plt.show()
        
        done = input('Press ‘ENTER’ to continue.')
        print()
        
        plt.close('all')
   
    # Funtion to open arduino connection
    def arduino_open(pressure_COM, temperature_COM):
        # Define baud rate
        baud_rate = 115200
        
        # Define timeout delay
        timeout = 1
        
        while True:
            
            # Try to connect to pressure and temperature arduinos
            try:
                # Define the pressure arduino input
                pressure_arduino = serial.Serial(pressure_COM, baud_rate, timeout=timeout)
                
                # Define the temperature arduino input
                temperature_arduino = serial.Serial(temperature_COM, baud_rate, timeout=timeout)
                
                # Break while loop
                break
            
            # If connection fails
            except serial.SerialException:
                pass
        
        # Delay for 2 second s allow for arduino connection
        time.sleep(2)
    
        return pressure_arduino, temperature_arduino
    
    # Function to close arduino connection
    def arduino_close(pressure_arduino, temperature_arduino):
        if pressure_arduino.is_open:
            pressure_arduino.close()
        if temperature_arduino.is_open:
            temperature_arduino.close()
    
    ######################### FUNCTION DEFINITIONS #########################
    
    ##### Function to reset pressure dictionary #####
    def reset_pressure_data():
        pressure_data = {
            'A0': [],
            'A1': [],
            'A2': [],
            'A3': [],
            'A4': [],
            'A5': [],
        }
        return pressure_data
    
    ##### Function to reset temperature list #####
    def reset_temperature_data():
        temperature_data = []

        return temperature_data
    
    ##### Function to convert resistance readings to pressure #####
    def pressure_calibration(average_pressure):
        average_pressure_cal = {
            'A0': [],
            'A1': [],
            'A2': [],
            'A3': [],
            'A4': [],
            'A5': [],
        }
        
        # Calibration values
            # offset
            # A1, B1, C1
            # A2, B2
            # A3, B3
        cal_values = {
            'A0': [2.9, 
                   72504.81, -44764.41, 10235.14, 
                   -552.44, 1862.82, 
                   -368.53, 1446.75],
            'A1': [3.02, 
                   14316.83, -20707.11, 10211.14,
                   -455.18, 2306.85,
                   -277.21, 1901.82],
            'A2': [2.97, 
                   19395.88, -23416.28, 10213.84,
                   -1034.35, 3313.65,
                   -571.38, 2243.63],
            'A3': [2.94, 
                   19585.98, -23448.86, 10213.88,
                   -532.86, 2282.79,
                   -643.52, 2533.45],
            'A4': [3.07, 
                   121390.62, -57848.52, 10248.19, 
                   -386.53, 1796.44, 
                   -471.51, 1991.49],
            'A5': [0.34, 
                   174773.28, -78236.92, 10268.52,
                   -300.90, 1637.16, 
                   -485.56, 2062.89],
        }

        # For all sensors
        for i in range(0, 6):
            resistance = average_pressure[f'A{i}']
            
            # Add offset back to resistance values
            resistance += cal_values[f'A{i}'][0]
            
            # If resistance is still negative after offset
            if resistance < 0:
                pressure = 12000
                
            # If resistance is less than 1, use quadratic function
            elif resistance < 1:
                pressure = (cal_values[f'A{i}'][1] * (resistance ** 2)) + (cal_values[f'A{i}'][2] * resistance) + cal_values[f'A{i}'][3]
            
            # If resistance is less than 10, use first natural log function
            elif resistance <= 10:
                pressure = (cal_values[f'A{i}'][4] * ln(resistance)) + cal_values[f'A{i}'][5]
            
            # If resistance is greater than 10, use first natural log function
            elif resistance <= 30:
                pressure = (cal_values[f'A{i}'][6] * ln(resistance)) + cal_values[f'A{i}'][7]
            
            else:
                pressure = 0
            
            average_pressure_cal[f'A{i}'].append(pressure)
            
        return average_pressure_cal
        
    ##### Function to calibrate the temperature readings #####
    def temperature_calibration(average_temperature):
        
        # Temperature calibration equation
        average_temperature_cal = ((1.1235 * average_temperature) - 2.0849)
        
        return average_temperature_cal
            
    ##### Function to close figure #####
    def on_close(event):
        print("Display window closed. Exiting gracefully.")
        plt.close('all')
        exit(0)

    ##### Function to show the pressure and temperature figure #####
    def pressure_temperature_plot(pressure, temperature):
        
        # Create first row with 0 pressure
        row1 = np.linspace(0, 0, num=300)

        # Create second row with 0 -> A0 reading -> A1 reading -> A2 reading -> 0 pressure
        row2 = np.concatenate((np.linspace(0, pressure['A0'], num=50),
                               np.linspace(pressure['A0'], pressure['A1'], num=100),
                               np.linspace(pressure['A1'], pressure['A2'], num=100),
                               np.linspace(pressure['A2'], 0, num=50)))
        
        # Create third row with 0 -> A3 reading -> A4 reading -> A5 reading -> 0 pressure
        row3 = np.concatenate((np.linspace(0, pressure['A3'], num=50),
                               np.linspace(pressure['A3'], pressure['A4'], num=100),
                               np.linspace(pressure['A4'], pressure['A5'], num=100),
                               np.linspace(pressure['A5'], 0, num=50)))
        
        # Create fourth row with 0 pressure
        row4 = np.linspace(0, 0, num=300)
        
        # Create three sections of zeros 
        section1 = np.zeros((50, 300)) # row1 -> row2
        section2 = np.zeros((100, 300)) # row2 -> row3
        section3 = np.zeros((50, 300)) # row3 -> row4

        # For the whole top row
        for i in range(0, 300):
            
            ## SECTION 1 ##
            temp = np.linspace(row1[i], row2[i], num=50) # Create a temporary line connecting row1 and row2
            temp_section1 = temp.reshape((50, 1)) # Reshape vertically
            section1[:, i] = temp_section1.flatten() # Create section1

            ## SECTION 2 ##
            temp = np.linspace(row2[i], row3[i], num=100) # Create a temporary line connecting row2 and row3
            temp_section2 = temp.reshape((100, 1)) # Reshape vertically
            section2[:, i] = temp_section2.flatten() # Create section2
            
            ## SECTION 3 ##
            temp = np.linspace(row3[i], row4[i], num=50) # Create a temporary line connecting row3 and row4
            temp_section3 = temp.reshape((50, 1)) # Reshape vertically
            section3[:, i] = temp_section3.flatten() # Create section3

        # Stact all sections vertically
        pressure_gradient = np.vstack((section1, section2, section3))
        
        # Turn on interactive mode
        plt.ion()
        
        # Clear current figure
        plt.clf()
        
        # Plot pressure gradient
        plt.imshow(pressure_gradient, cmap='turbo', interpolation='lanczos', vmin=0, vmax=10000)
        
        # Add color bar
        plt.colorbar()
        
        # Add applied pressure and units title
        plt.title('Applied Pressure (Pa)')
        
        # Delete ticks
        plt.xticks([])
        plt.yticks([])
        
        # Add temperature reading to the plot
        plt.text(0, 220, f'Temperature: {temperature:.2f} °C', bbox=dict(facecolor='white', edgecolor='white'))
        
        # Show the figure
        plt.draw()
        
        # Pause for a short period of time
        plt.pause(0.1)
        
        # Close the figure
        plt.gcf().canvas.mpl_connect('close_event', on_close)
    
    ######################### CALIBRATION #########################
    
    COM_1 = 'COM4'
    COM_2 = 'COM5'
    
    display_instructions('calibration.png')
    
    # Open arduino connection
    pressure_arduino, temperature_arduino = arduino_open(COM_1, COM_2)
            
    # Print 'Calibrating...' for user
    print('Calibrating...')
    
    # Define calibration counter
    count = 0
    
    # Define the number of calibration points
    calibration_count = 10

    # Define prssure offset dictionary
    pressure_offset = {
        'A0': [],
        'A1': [],
        'A2': [],
        'A3': [],
        'A4': [],
        'A5': [],
    }

    # Define temperature offset list
    temperature_offset = []

    # Try to run until keyboard interruption
    try:
        
        # Try to decode 'utf-8'
        try:
            # Continue calibration until calibration count is reached
            while count < calibration_count:
                
                # Read each line from the pressure arduino
                pressure_line = pressure_arduino.readline().decode('utf-8').strip()
                
                # Separate pressure line by tabs 
                pressure_output = pressure_line.split('\t')
                
                # Read each line from the temperature arduino
                temperature_line = temperature_arduino.readline().decode('utf-8').strip()
                
                # Separate temperature line by tabs 
                temperature_output = temperature_line.split('\t')
    
                # If the pressure output contains readings from all 6 pressure sensors
                if len(pressure_output) == 6:
                    
                    # For every pressure output reading
                    for i in range(0, 6):
                        
                        # If the reading is infinity (no pressure)
                        if pressure_output[i] == 'inf':
                            
                            # Report zero pressure
                            pressure_offset[f'A{i}'].append(0)
                            
                        # Otherwise interprut the reading using the calibration curve
                        else:
                            
                            # Add the reading to the pressure offset dictionary
                            pressure_offset[f'A{i}'].append(float(pressure_output[i]))
                            
                # Try to convert the temperature output reading to a number
                try:
                    
                    # Add the reading to the temperature offset list
                    temperature_offset.append(float(temperature_output[0]))
                
                # If the temperature reading can not be converted
                except ValueError:
                    pass
                                 
                # Add 1 to the count
                count += 1
                
                # Calculate the calibration precentage
                percentage = (count / calibration_count) * 100
                
                # Print the calibration precentage
                print(f'{round(percentage)}%')
        
        # If there is a decode error
        except UnicodeDecodeError:
            pass
        
        # Average all of the readings for each sensor in the pressure offset dictionary
        average_pressure_offset = {sensor: np.mean(data) for sensor, data in pressure_offset.items()}
        
        # Average all of the readings in the temperature list
        average_temperature_offset = np.mean(temperature_offset)
        
        # Calibrate the average pressure reading
        average_pressure_offset_cal = pressure_calibration(average_pressure_offset)        
        
        # Calibrate the average temperature reading
        average_temperature_offset_cal = temperature_calibration(average_temperature_offset)

        
    # If there is a keyboard interruption  
    except KeyboardInterrupt:
        
        # Print 'Exiting gracefully.' for user
        print('Exiting gracefully.')
        pass

    # Print room temperature and exit instructions for user
    print(f'Calibration complete! Room Temperature: {average_temperature_offset_cal:.2f} °C.\n\nTemperature readings are most accurate 30 seconds after application.\nPress ‘CTRL + C’ or ‘CMD + C’ at any point to exit program.\n')
    
    # Close arduino connection
    arduino_close(pressure_arduino, temperature_arduino)
    
    display_instructions('application.png')

    ######################### PRESSURE AND TEMPERATURE FIGURE #########################
    
    # Open arduino connection
    pressure_arduino, temperature_arduino = arduino_open(COM_1, COM_2)  
    
    csv_data = {
        'A': [],
        'B': [],
        'C': [],
        'D': [],
        'E': [],
        'F': [],
        'G': []
    }
    
    # Reset pressure dictionary
    pressure_data = reset_pressure_data()
    
    # Reset temperature dictionary
    temperature_data = reset_temperature_data()

    # Run until keyboard interruption
    try:
        
        # Run indefinitely
        while True:
            
            # Try to decode 'utf-8'
            try:
                # Read each line from the pressure arduino
                pressure_line = pressure_arduino.readline().decode('utf-8').strip()
                
                # Separate pressure line by tabs 
                pressure_output = pressure_line.split('\t')
                
                # Read each line from the temperature arduino
                temperature_line = temperature_arduino.readline().decode('utf-8').strip()
                
                # Separate temperature line by tabs 
                temperature_output = temperature_line.split('\t')
                            
                # If the average count is less than 5
                if len(pressure_data['A0']) < 5 or len(temperature_data) < 5:
                    
                    # If the pressure output contains readings from all 6 pressure sensors
                    if len(pressure_output) == 6:
                        
                        # For every pressure output reading
                        for i in range(0, 6):
                            
                            # If the reading is infinity (no pressure)
                            if pressure_output[i] == 'inf':
                                
                                # Report zero pressure
                                pressure_data[f'A{i}'].append(0)
                                
                            # Otherwise interprut the reading using the calibration curve
                            else:
                                
                                # Add the reading to the pressure offset dictionary
                                pressure_data[f'A{i}'].append(float(pressure_output[i]) - average_pressure_offset_cal[f'A{i}'][0])
                                pressure_data[f'A{i}'].append(float(pressure_output[i]))
                                
                    # Try to convert the temperature output reading to a number
                    try:
                        
                        # Add the reading to the temperature offset list
                        temperature_data.append(float(temperature_output[0]))
                        
                    # If the temperature reading can not be converted
                    except ValueError:
                        pass
    
                else:
                    # Average all of the readings for each sensor in the pressure data dictionary   
                    average_pressure = {sensor: np.mean(data) for sensor, data in pressure_data.items()}
                    
                    # Average all of the readings in the temperature list
                    average_temperature = np.mean(temperature_data)
                    
                    # Calibrate the average pressure reading
                    average_pressure_cal = pressure_calibration(average_pressure)
                    
                    # Calibrate the average temperature reading
                    average_temperature_cal = temperature_calibration(average_temperature)
                    
                    # Show the pressure and temperature figure
                    pressure_temperature_plot(average_pressure_cal, average_temperature_cal)
                    
                    # Add the average pressure and temperature to the csv
                    for i in range(0, 6):
                        columns = ['A', 'B', 'C', 'D', 'E', 'F']
                        csv_data[f'{columns[i]}'].append(average_pressure_cal[f'A{i}'][0])
                    csv_data['G'].append(average_temperature_cal)

                    # Reset pressure dictionary
                    pressure_data = reset_pressure_data()
                    
                    # Reset temperature dictionary
                    temperature_data = reset_temperature_data()
            
            # If there is a decode error
            except UnicodeDecodeError:
                pass
    
    # If there is a keyboard interruption  
    except KeyboardInterrupt:
        
        name = input('Name the PyraBoto data file:')
        print()
        
        # Clear figure
        plt.clf()
        
        # Display the thank you message
        plt.imshow(mpimg.imread('thank_you.png'))
        
        # Hide the axes of the plot
        plt.axis('off')
        
        # Improve figure quality        
        plt.rcParams['figure.dpi'] = 500
        
        
        
        # Show the plot
        plt.show()
        
        # Create a excel workbook        
        workbook = xlsxwriter.Workbook(f'{name}.xlsx')
         
        # Add a worksheet to the workbook
        worksheet = workbook.add_worksheet()
         
        # Add headings
        worksheet.write('A1', 'A0 Pressure (Pa)')
        worksheet.write('B1', 'A1 Pressure (Pa)')
        worksheet.write('C1', 'A2 Pressure (Pa)')
        worksheet.write('D1', 'A3 Pressure (Pa)')
        worksheet.write('E1', 'A4 Pressure (Pa)')
        worksheet.write('F1', 'A5 Pressure (Pa)')
        worksheet.write('G1', 'Temperature (°C)')
        
        # Add the data to the workbook
        for column in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            for row in range(len(csv_data['A'])):
                worksheet.write(f'{column}{row + 2}', csv_data[column][row])

        # Close the workbook
        workbook.close()
        
        print("Exiting gracefully.")
        
        # Delay for 5 seconds
        time.sleep(5)
        
        pass
    
    # Close arduino serials
    finally:
        # Close arduino connection
        arduino_close(pressure_arduino, temperature_arduino)

if __name__ == "__main__":
    main()
