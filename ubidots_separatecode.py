import requests
import json
import time

def send_to_ubidots_r1(value):
    url = "https://industrial.api.ubidots.com/api/v1.6/devices/iot"
    payload = { "r1": value }
    headers = { "Content-Type": "application/json", "X-Auth-Token": "BBUS-hLmxzEtc7efjDFE0AxufTAJVXyISXb" }

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        print(f"Data sent successfully: {value}")
    else:
        print(f"Failed to send data. Status code: {response.status_code}")
        print(f"Response content: {response.content}")
        
def send_to_ubidots_r2(value):
    url = "https://industrial.api.ubidots.com/api/v1.6/devices/iot"
    payload = { "r2": value }
    headers = { "Content-Type": "application/json", "X-Auth-Token": "BBUS-hLmxzEtc7efjDFE0AxufTAJVXyISXb" }

    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        print(f"Data sent successfully: {value}")
    else:
        print(f"Failed to send data. Status code: {response.status_code}")
        print(f"Response content: {response.content}")

# Example usage:
# Replace this with your own logic to get the value (1 or 0)
value = 0  # or 0

send_to_ubidots_r1(value)
send_to_ubidots_r2(value)

# # Send data every 5 seconds (adjust as needed)
# time.sleep(5)
# value= 0

# send_to_ubidots(value)

# time.sleep(5)
