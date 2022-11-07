#include <Wire.h>
//#include <WireExt.h>

#define D6T_addr 0x0A // Address of OMRON D6T is 0x0A in hex
#define D6T_cmd 0x4C // Standard command is 4C in hex
 
int rbuf[19]; // Actual raw data is coming in as 35 bytes. Hence the needed for WireExt so that we can handle more than 32 bytes
int tdata[8]; // The data comming from the sensor is 16 elements, in a 16x1 array
float t_PTAT;

void setup()
{
  Wire.begin();
  Serial.begin(9600);

  pinMode(13, OUTPUT);
}
 
void loop()
{
  int i;
      // Begin a regular i2c transmission. Send the device address and then send it a command.
      Wire.beginTransmission(D6T_addr);
      Wire.write(D6T_cmd);
      Wire.endTransmission();
      delay(100);
      i=0;
      
      // This is where things are handled differently. Omron D6T output data is 35 bytes and there is a limit here on what Wire can receive. We use WireExt to read the output data 1 piece at a time. We store each byte as an element in rbuf.
      //for (int cnt = 0; cnt < 7; cnt++){
        Serial.print("hai");
        Wire.requestFrom(D6T_addr, 19);
  //for (int j = 0; cnt < 5; j++){
   //  Serial.print("bhavya");
        while(Wire.available()){
        Serial.print("hello");
        Serial.print(rbuf[i]);
        rbuf[i] = Wire.read();
        Serial.print(rbuf[i]);
        i++;
   
  }
  delay(100);
  Wire.endTransmission();

  
  t_PTAT = (rbuf[0]+(rbuf[1]<<8))*0.1; 
        
        // Calculate the temperature values: add the low temp and the bit shifted high value together. Multiply by 0.1
        for (i = 0; i < 8; i++) {
          tdata[i]=(rbuf[(i*2+2)]+(rbuf[(i*2+3)]<<8))*0.1;
          
        }        
      // Use a for loop to output the data. We can copy this from serial monitor and save as a CSV
      for (i=0; i<8; i++) {
        Serial.print(tdata[i]);
        Serial.print(",");
      }
      Serial.print("\n");
      
          delay(1000);          
}
