// Define analog pins for pressure sensors
int FSR_pins[] = {A0, A1, A2, A3, A4, A5};

int avg_size = 1000; // number of analog readings to average
float R_0 = 10000.0; // known resistor value in [Ohms]
float Vcc = 3.3; // supply voltage

void setup(void) {
  Serial.begin(115200);
}

void loop() {

  // Read pressure sensor data
  for (int sensorIndex = 0; sensorIndex < 6; sensorIndex++) {
    float sum_val = 0.0; // variable for storing sum used for averaging
    float R_FSR;
    
    for (int ii = 0; ii < avg_size; ii++) {
      sum_val += (analogRead(FSR_pins[sensorIndex]) / 1023.0) * 5.0; // sum the 10-bit ADC ratio
    }
    sum_val /= avg_size; // take average
    
    R_FSR = (R_0 / 1000.0) * ((Vcc / sum_val) - 1.0); // calculate actual FSR resistance

    // Print the sensor index, its corresponding name, and the resistance to the Serial Plotter
    Serial.print(R_FSR);
    Serial.print('\t'); // Add a tab to separate values
  }

  Serial.println(); // Print a newline to start a new line

  delay(0); // Adjust delay as needed
}