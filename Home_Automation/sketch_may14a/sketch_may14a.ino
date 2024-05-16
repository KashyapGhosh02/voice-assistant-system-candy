#include <WiFi.h>
#include <PubSubClient.h>

/**************
 * Define Constants
 **************/
#define TOKEN "BBUS-hLmxzEtc7efjDFE0AxufTAJVXyISXb" // Your Ubidots TOKEN
#define WIFINAME "DEMON"                    // Your SSID
#define WIFIPASS "a1b2c34d5e"                 // Your Wifi Pass
#define MQTT_CLIENT_NAME "RandomName"

#define VARIABLE_LABEL1 "r1"
#define VARIABLE_LABEL2 "r2"
#define DEVICE_LABEL "iot"
#define RELAY_PIN1 2
#define RELAY_PIN2 4
char mqttBroker[] = "things.ubidots.com";
char payload[100];
char topic[150];
char topicSubscribe1[100];
char topicSubscribe2[100];

WiFiClient ubidots;
PubSubClient client(ubidots);

void callback(char *topic, byte *payload, unsigned int length)
{
  char p[length + 1];
  memcpy(p, payload, length);
  p[length] = '\0'; // Null-terminate the string
  String message(p);

  //code to extrct r1 or r2
  String str = String(topic);
  int startIndex = str.indexOf("/r");
  int endIndex = str.indexOf("/", startIndex + 1);
  String value = str.substring(startIndex + 1, endIndex);// r1 or r2
  //------------------------------------
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
  }
  Serial.println();
//---------------------------------------

  if(value=="r1")
  {
  if (message == "1.0")
  {
    digitalWrite(RELAY_PIN1, LOW);
    Serial.println("turned on");
  }
  else
  {
    digitalWrite(RELAY_PIN1, HIGH);
    Serial.println("turned off");
  }
  }

  if(value=="r2")
  {
  if (message == "1.0")
  {
    digitalWrite(RELAY_PIN2, LOW);
    Serial.println("turned on");
  }
  else
  {
    digitalWrite(RELAY_PIN2, HIGH);
    Serial.println("turned off");
  }
  }
}

void reconnect()
{
  // Loop until we are connected
  while (!client.connected())
  {
    Serial.println("Attempting MQTT connection ...");
    if (client.connect(MQTT_CLIENT_NAME, TOKEN, ""))
    {
      Serial.println("Connected");
      client.subscribe(topicSubscribe1);
      client.subscribe(topicSubscribe2);
    }
    else
    {
      Serial.print("Failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 2 seconds");
      delay(2000);
    }
  }
}

void setup()
{
  Serial.begin(115200);
  WiFi.begin(WIFINAME, WIFIPASS);
  pinMode(RELAY_PIN1, OUTPUT);
  pinMode(RELAY_PIN2,OUTPUT);
  Serial.print("Wait for WiFi ...");
  while (WiFi.status() != WL_CONNECTED)
  {
    Serial.print(".");
    delay(500);
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address:");
  Serial.println(WiFi.localIP());
  client.setServer(mqttBroker, 1883);
  client.setCallback(callback);
  sprintf(topicSubscribe1, "/v1.6/devices/%s/%s/lv", DEVICE_LABEL, VARIABLE_LABEL1);
  client.subscribe(topicSubscribe1);
  sprintf(topicSubscribe2, "/v1.6/devices/%s/%s/lv", DEVICE_LABEL, VARIABLE_LABEL2);
  client.subscribe(topicSubscribe2);
}

void loop()
{
  if (!client.connected())
  {
    client.subscribe(topicSubscribe1);
    client.subscribe(topicSubscribe2);
    reconnect();
  }

  client.loop();
  delay(1000);
}
