#include <L298N.h>
#include "CytronMotorDriver.h"
#include <Firmata.h>

// Pin definition
const unsigned int IN1 = 7;
const unsigned int IN2 = 8;
const unsigned int EN = 9;
double ch3=1;
double ch5=1;
double ch6=1;
double ch5_old=1000;
//const unsigned int DIR_PIN = 13;
//const unsigned int PWM_PIN = 11;
String received_data = "";
int brake_active = 0;
int maxspeed = 255;
float currentspeed=0;
float increment = 40;
int go_count = 0;
int max_go_count = 0;

L298N motor(EN, IN1, IN2);

CytronMD brake_motor(PWM_DIR, 11, 13);  // PWM = Pin 11, DIR = Pin 13.

void setup()
{ 
  pinMode(2,INPUT);
  pinMode(3,INPUT);
  pinMode(4,INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
//  Firmata.begin();
  

  while (!Serial)
  {
  }

  motor.setSpeed(0);
}

void loop() {
  ch6 = pulseIn(2,HIGH);
  Serial.print(ch6);
  if (ch6 > 1500){
    Serial.println("Autonomous Mode");
    
    if (ch3<990)
      if (Serial.available()) {
          received_data = Serial.readStringUntil('\n');
          
          if (received_data.equals("stop")) {
              
              go_count = 0;
              
              Serial.println("stopping vehicle");
              brake_motor.setSpeed(255);
  //            delay(2000);
  //            brake_motor.setSpeed(0);
              brake_active = 1;
  
              motor.setSpeed(0);
              motor.forward();
              currentspeed = 0;
                            
  //            digitalWrite(LED_BUILTIN, HIGH); // Turn the LED on
  //            delay(500); // Wait for 500 milliseconds
  //
  //            digitalWrite(LED_BUILTIN, LOW); // Turn the LED off
  //            delay(500); // Wait for 500 milliseconds
          }
            else{
//          if (received_data.equals("stoop")) {
            go_count++;
            if (go_count > max_go_count){
              if (brake_active == 1){
                Serial.println("releasing brake");
                brake_motor.setSpeed(-255);
                delay(500);
                brake_motor.setSpeed(0);
                brake_active = 0;
                }
              if (currentspeed < maxspeed){
                currentspeed+=increment;
                if (currentspeed > maxspeed){
                  currentspeed = maxspeed;
              }
            }
//                currentspeed = 250;
//                Serial.println(currentspeed);
                motor.setSpeed(currentspeed);
                motor.forward();
                Serial.print("Speed = ");
                Serial.println(motor.getSpeed());
              }
            }
          }
  }
///////////////////////////////////////////////////////////////////////////////////////
  else{
    // Remote Control Code

      Serial.print("RC Mode: ");
      
      digitalWrite(2,HIGH);
      ch3 = pulseIn(3,HIGH);
      if (ch3<990){
        motor.stop();
      }
      else{
        int speed = mapValue(ch3, 1000, 2000, 0, 255);
        if (speed > 0 && speed <= 255) {
            // Set the speed and move forward
            motor.setSpeed(speed);
            motor.forward();
          }
      }
    
      ch5 = pulseIn(4,HIGH);
      if (ch5 > 1800){
        brake_motor.setSpeed(255);
        ch5_old = ch5;
      }
      ch5 = pulseIn(4,HIGH);
      if (ch5 < 1800){
        if (ch5_old > 1800){
          Serial.print(" RC ch5 value = ");
          Serial.print(ch5);
          Serial.print(" RC ch5_old value = ");
          Serial.print(ch5_old);
          brake_motor.setSpeed(-255);
          delay(1000);
          brake_motor.setSpeed(0);
          ch5_old = ch5;
        }
      }
    
    
      Serial.print("RC ch3 value = ");
      Serial.print(ch3);
      Serial.print(" RC ch5 value = ");
      Serial.print(ch5);
      Serial.print(" RC ch5_old value = ");
      Serial.print(ch5_old);
      Serial.print(" Motor is moving = ");
      Serial.print(motor.isMoving());
      Serial.print(" at speed = ");
      Serial.println(motor.getSpeed());
  }
}

int mapValue(int value, int fromLow, int fromHigh, int toLow, int toHigh) 
{
    float scaledValue = float(value - fromLow) / float(fromHigh - fromLow);
    return int(scaledValue * (toHigh - toLow) + toLow);
}
