#include <WiFi.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>

const char* ssid     = "SSID";
const char* password = "PASSWORD";

const int udpPort = 4210;  
WiFiUDP udp;

Servo myServo;
const int servoPin = 21;

void setup() {
  Serial.begin(115200);

  // Allow servo PWM on this pin
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  myServo.setPeriodHertz(50);    // Standard 50 Hz servo frequency
  myServo.attach(servoPin, 500, 2400);  // Min/max pulse lengths

  myServo.write(90);  // Initial position

  // Connect to WiFi
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  udp.begin(udpPort);
  Serial.print("Listening for UDP on port ");
  Serial.println(udpPort);
}

void loop() {
  int packetSize = udp.parsePacket();
  if (packetSize) {
    char incoming[255];
    int len = udp.read(incoming, 255);
    if (len > 0) incoming[len] = '\0';

    Serial.print("UDP Packet: ");
    Serial.println(incoming);

    // Example: Move servo on any UDP packet
    myServo.write(0);
    delay(2000);
    myServo.write(90);
  }
}
