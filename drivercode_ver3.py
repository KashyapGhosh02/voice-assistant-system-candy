#import necessary modules
import os
import subprocess
# from datetime import datetime
import speech_recognition as sr
import wikipedia
import bs4
from bs4 import BeautifulSoup
import requests
import datetime
import time
import random

import RPi.GPIO as GPIO
import time
import warnings
warnings.filterwarnings("ignore")
#------------------------------------------------

# Set up GPIO using BCM numbering
GPIO.setmode(GPIO.BCM)

# Set GPIO NUMBER as output
green_led_pin = 26
yellow_led_pin=20
red_led_pin=21
GPIO.setup(green_led_pin, GPIO.OUT)
GPIO.setup(yellow_led_pin, GPIO.OUT)
GPIO.setup(red_led_pin, GPIO.OUT)

GPIO.output(green_led_pin, GPIO.LOW)
GPIO.output(yellow_led_pin, GPIO.LOW)
GPIO.output(red_led_pin, GPIO.HIGH)
#------------------------------
#import custom-made modules 
import voice_greetings
import reminder_property
from wolframalpha_rename import wolframalpha_result
import ubidots_separatecode
#import prediction_func 
#------------------------------
# for pc :-
# #Initializing pyttsx3
# import pyttsx3
# listening = True
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)

# def say(audio):
#     engine.say(audio)
#     engine.runAndWait()
    
def say(text, language='en-US'):
    # Define the command to generate speech using pico2wave
    GPIO.output(yellow_led_pin, GPIO.HIGH)
    pico_command = f'pico2wave -l {language} -w speech.wav "{text}"'
    
    # Execute the pico2wave command
    os.system(pico_command)
    
    # Play the generated speech using aplay
    os.system('aplay speech.wav')
    
    # Clean up the temporary speech file
    os.remove('speech.wav') 
    GPIO.output(yellow_led_pin, GPIO.LOW)
#----------------speech-recognition--------------
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        GPIO.output(green_led_pin, GPIO.HIGH)
        r.pause_threshold = 1
        r.energy_threshold=3000
        try:
            audio = r.listen(source, timeout=3)
            GPIO.output(green_led_pin, GPIO.LOW)
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")
            return query.lower()
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return "none" # End of takeCommand() function
    
def wishMe(query): 
    #list_int_resp=prediction_func.predict_intent_with_nltk(text=query)
    list_int_resp=voice_greetings.predict_intent(text=query)
    return list_int_resp

   
    

def google_search(search_text):
    result = ''
    search_data = search_text
    if "who is" in search_data or "who are" in search_data:
        search_data = search_data.split(" ")[2:]
        search_data = " ".join(search_data)
        try:
            result = wikipedia.summary(search_data, sentences=2)
            return result
        except Exception as e:
            pass
    else:
        url = "https://www.google.co.in/search?q=" + search_data
        try:
            search_result = requests.get(url).text
            soup = BeautifulSoup(search_result, 'html.parser')
            result_div = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd')
            result = result_div[0].text
            return result
        
        except Exception as e:
            pass
    return("sorry could not find any information")

       
          
ask_response = [
    "Is there anything else on your mind?",
    "Do you need assistance with anything else?",
    "Would you like to discuss anything further?",
    "Are there any other topics you'd like to explore?",
    "Feel free to ask me anything else.",
    "Is there something specific you'd like to know more about?",
]
random_response = random.choice(ask_response)

def start():
    try:
        while True:
            curr_time = datetime.datetime.now().strftime("%H:%M:00")
            print(curr_time)
            reminder_property.GetReminder(curr_time)    
            query = takeCommand()
            #if query is None:
            #    continue  # Continue listening if no query is obtained
            if "candy" in query.lower():
                list_intent_response = wishMe(query)
                say(list_intent_response[1])
                while True:
                    query = takeCommand()
                    list_intent_response = wishMe(query)

                    if list_intent_response[0] == "goodbye" or list_intent_response[0]=="gratitude" :
                        say(list_intent_response[1])
                        break
                    elif "no" in query or query.lower() =="none" and list_intent_response[0]=="nothing" :
                        say(list_intent_response[1])
                        break
                    else:
                        get_result(query, list_intent_response)
                    say(random_response)   
    except Exception as e:
        GPIO.cleanup()
        print(e)


def get_result(query,list_intent_response):
    
    if query.lower()=="none" and list_intent_response[0]=="nothing":      
        say("No instructions found")
    
    elif "add" in query or "set" in query and "reminder" in query:
        say("Add description")
        desc_1=takeCommand()
        if desc_1=="none": 
            say("Did not get any Description")
            return
        say("Add time")
        time_1=takeCommand()
        reminder_property.SetReminder(desc_1,time_1)
    
    elif "delete" in query and "reminder" in query:
        say("Give the time of the reminder to be deleted")
        time_1=takeCommand()
        reminder_property.DeleteReminder(time_1)

    elif "time now" in query or "time in " in query or "time at " in query  or "date in "in query or "today's date " in  query or "weather in " in query or "weather at " in query or "temperature in "  in query or "temperature at "  in query: # for time date and weather related info
        x = datetime.datetime.now()
        if "date" in query: #Only local Date
            say("Date is "+x.strftime("%A ")+x.strftime("%d ")+x.strftime("%B ")+x.strftime("%Y ")) # Date
        if "time" in query: #time can be searched for local and international zones   
            url = "https://google.com/search?q=" + query
            request_result = requests.get( url )
            soup = bs4.BeautifulSoup( request_result.text  , "html.parser" ) 
            time = soup.find( "div" , class_='BNeawe' ).text  
            say("time is "+time ) 

        if "weather" or "temperature" in query:
            url = "https://www.google.com/search?q="+query
            html = requests.get(url).content
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
            str1 = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text
            data = str1.split('\n')
            sky = data[1]
            say("Temperature is "+ temp)
            say("Sky Description:  "+ sky)
        

          
    elif 'wikipedia' in query:     # Wikipedia search
        say('Searching Wikipedia...')
        query = query.replace("wikipedia", "")
        results = wikipedia.summary(query, sentences=4)
        say("According to Wikipedia")
        say(results)


    elif list_intent_response[0]=="math" and "math" in query:
        result=wolframalpha_result(question=query)
        say(result)
        return
    elif "turn on" in query:
        if "socket one" in query or "socket 1" in query:
            say("turning on the socket 1")
            ubidots_separatecode.send_to_ubidots_r1(1)
        if "socket two" in query or "socket 2" in query:
            say("turning on the socket 2")
            ubidots_separatecode.send_to_ubidots_r2(1)
        if "both" in query:
            say("turning on the both the socket")
            ubidots_separatecode.send_to_ubidots_r1(1)
            ubidots_separatecode.send_to_ubidots_r2(1)
    elif "turn off " in query:
        if "socket one" in query  or "socket 1" in query:
            say("turning off the socket one")
            ubidots_separatecode.send_to_ubidots_r1(0)
        if "socket two" in query or "socket 2" in query:
            say("turning off the socket two")
            ubidots_separatecode.send_to_ubidots_r2(0)
        if "both" in query:
            say("turning off the both the socket")
            ubidots_separatecode.send_to_ubidots_r1(0)
            ubidots_separatecode.send_to_ubidots_r2(0)
    elif "reboot" in query:
        say("rebooting the system")
        os.system("sudo reboot")
    elif "shutdown" in query:
        say("shutting down the system")
        os.system("sudo shutdown now")
    else:
        results = google_search(query) #google serach
        print(results)
        say(results)
    
            
if __name__ == "__main__":
    try:
        print("starting the code ")
        GPIO.output(green_led_pin, GPIO.LOW)
        GPIO.output(red_led_pin, GPIO.LOW)
        start()
    except Exception as e:
        GPIO.cleanup()
        print(e)
          
            

          
          


