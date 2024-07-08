#include <Servo.h>

Servo servo;
int servoPin = 9;  // Change this to the pin number connected to your servo

void setup() {
  servo.attach(servoPin);
  servo.write(90);  // Start with the servo at the 90-degree position
  Serial.begin(115200);
}

void loop() {
  if (Serial.available() > 0) {
    String angleString = Serial.readStringUntil('\n');
    int angle = angleString.toInt();
    servo.write(angle);
 // For debugging, print the angle received
  }
} 
