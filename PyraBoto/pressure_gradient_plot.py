#!/usr/bin/env python
import serial
import time
import matplotlib.pyplot as plt
import numpy as np
import serial.tools.list_ports
from random import randint

def main():
    # COM_num = input('Enter the number of the COM port you are using: ')
    # COM_num = 3
    ports = serial.tools.list_ports.comports()
    if ports:
        COM_num = ports[0].device
        print(f'Using COM port: {COM_num}')
    else:
        print('No COM ports found. Please connect your device.')
        return

    baud_rate = 115200
    timeout = 1
    # ser = serial.Serial(f'COM{COM_num.strip()}', baud_rate, timeout=timeout)
    ser = serial.Serial(COM_num, baud_rate, timeout=timeout)

    time.sleep(2)

    calibration = input('Press enter to begin the calibration process. Remove all pressure and temperature sources.')
    print('Calibrating...')
    count = 0

    pressure_offset = {
        'A0': [],
        'A1': [],
        'A2': [],
        'A3': [],
        'A4': [],
        'A5': [],
    }

    temperature_offset = []

    try:
        total_count = 50
        while count < 50:
            line = ser.readline().decode('utf-8').strip()
            data = line.split('\t')

            if len(data) == 5:
                for i in range(0, 4):
                    if data[i] == 'inf':
                        pressure_offset[f'A{i}'].append(0)
                    else:
                        if float(data[i]) > 100:
                            pressure_offset[f'A{i}'].append(0)
                        else:
                            pressure_offset[f'A{i}'].append(float(data[i]))
                temperature_offset.append(float(data[4]))
                pressure_offset['A4'].append(0)
                pressure_offset['A5'].append(0)
            count += 1
            percentage = (count / total_count) * 100
            print(f'{round(percentage)}%')

        average_pressure_offset = {sensor: np.mean(data) for sensor, data in pressure_offset.items()}
        average_temperature_offset = np.mean(temperature_offset)
        
        print(f'Pressure Calibration: {average_pressure_offset}')

    except KeyboardInterrupt:
        print("Exiting gracefully.")
        pass

    print(f'Calibration complete! Room Temperature: {average_temperature_offset:.2f} °C. \nPress CTRL + C or CMD + C to exit program')

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
    
    def on_close(event):
        print("Display window closed. Exiting gracefully.")
        plt.close('all')
        exit(0)

    def reset_temperature_data():
        temperature_data = []

        return temperature_data

    def pressure_temperature_plot(pressure, temperature):
        
        print(f'Pressure: {pressure}')
        print(f'Temperature: {temperature}')
        
        row1 = np.linspace(0, 0, num=300)

        row2 = np.concatenate((np.linspace(0, pressure['A0'], num=50),
                               np.linspace(pressure['A0'], pressure['A1'], num=100),
                               np.linspace(pressure['A1'], pressure['A2'], num=100),
                               np.linspace(pressure['A2'], 0, num=50)))

        row3 = np.concatenate((np.linspace(0, pressure['A3'], num=50),
                               np.linspace(pressure['A3'], pressure['A4'], num=100),
                               np.linspace(pressure['A4'], pressure['A5'], num=100),
                               np.linspace(pressure['A5'], 0, num=50)))

        row4 = np.linspace(0, 0, num=300)

        section1 = np.zeros((50, 300))
        section2 = np.zeros((100, 300))
        section3 = np.zeros((50, 300))

        for i in range(0, 300):
            temp = np.linspace(row1[i], row2[i], num=50)
            temp_section1 = temp.reshape((50, 1))
            section1[:, i] = temp_section1.flatten()

            temp = np.linspace(row2[i], row3[i], num=100)
            temp_section2 = temp.reshape((100, 1))
            section2[:, i] = temp_section2.flatten()

            temp = np.linspace(row3[i], row4[i], num=50)
            temp_section3 = temp.reshape((50, 1))
            section3[:, i] = temp_section3.flatten()

        pressure_gradient = np.vstack((section1, section2, section3))
        
        # turn on interactive mode and clear the figure
        plt.ion()
        plt.clf()
        plt.imshow(pressure_gradient, cmap='turbo', interpolation='lanczos', vmin=-10, vmax=10)
        plt.colorbar()  # Add colorbar on the right side
        plt.title('Applied Pressure (Pa)')
        plt.xticks([])
        plt.yticks([])
        plt.text(0, 220, f'Temperature: {temperature:.2f} °C', bbox=dict(facecolor='white', edgecolor='white'))
        plt.draw()
        plt.pause(0.1) # pause for a short period of time
        plt.gcf().canvas.mpl_connect('close_event', on_close)

    pressure_data = reset_pressure_data()
    temperature_data = reset_temperature_data()

    try:
        while True:
            # Read a line from the serial port
            line = ser.readline().decode('utf-8').strip()

            # Split the line into sensor names and corresponding resistance values
            data = line.split('\t')

            if len(data) == 5:
                if len(pressure_data['A0']) < 2:
                    for i in range(0, 4):
                        if data[i] == 'inf':
                            pressure_data[f'A{i}'].append(0) 
                        else:   
                            if float(data[i]) > 40:
                                pressure_data[f'A{i}'].append(0)
                            else:
                                pressure_data[f'A{i}'].append(float(data[i]) - average_pressure_offset[f'A{i}'])
                    pressure_data['A4'].append(0)
                    pressure_data['A5'].append(0)
                    temperature_data.append(float(data[4]))

                else:
                    
                    average_pressure = {sensor: np.mean(data) for sensor, data in pressure_data.items()}
                    average_temperature = np.mean(temperature_data)

                    pressure_temperature_plot(average_pressure, average_temperature)

                    pressure_data = reset_pressure_data()
                    temperature_data = reset_temperature_data()

    except KeyboardInterrupt:
        print("Exiting gracefully.")
        pass

    finally:
        # Close the serial connection
        if ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
