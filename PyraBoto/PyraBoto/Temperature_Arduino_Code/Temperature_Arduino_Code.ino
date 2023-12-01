#include <Wire.h>
#include <Adafruit_TMP117.h>
#include <Adafruit_Sensor.h>

#define USE_TMP117 // Define this flag to include TMP117 code

#ifdef USE_TMP117
Adafruit_TMP117 tmp117;
#endif

void setup(void) {
  Serial.begin(115200);
  
#ifdef USE_TMP117
  if (!tmp117.begin()) {
  }
#endif
}

void loop() {

#ifdef USE_TMP117
  // Read TMP117 data
  sensors_event_t temp; // create an empty event to be filled
  tmp117.getEvent(&temp); //fill the empty event object with the current measurements
  Serial.print(temp.temperature);
  Serial.print('\t');
#endif

  Serial.println(); // Print a newline to start a new line

  delay(1000); // Adjust delay as needed
}