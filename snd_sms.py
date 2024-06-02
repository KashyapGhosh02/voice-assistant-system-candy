import requests

# your API secret from (Tools -> API Keys) page
apiSecret = "b8e17ba03e2ae114a023a9d2a44341cc440fb0c7"
# msg="hi your voice assistant had a message for you : "
def smssnd(msg):

    message = {
        "secret": apiSecret,
        "mode": "devices",
        "device": "00000000-0000-0000-4ae4-ec5675c45353",
        "sim": 1,
        "priority": 1,
        "phone": "+918670457072",
        "message": "hi your voice assistant had a message for you :"+ msg
    }

    r = requests.post(url = "https://www.cloud.smschef.com/api/send/sms", params = message)
    
    # do something with response object
    result = r.json()