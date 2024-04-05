# import module 
import os
import sqlite3
from datetime import datetime
# conn=sqlite3.connect('reminder.db')
# c=conn.cursor()
# c.execute("""CREATE TABLE VoiceReminder(
#           desc text,
#           time text
# )""")
# create Table------------------------- Run only once
#---------------------------------------


    
# for rasp :- 
# initialize pico
def say(text, language='en-US'):
    # Define the command to generate speech using pico2wave
    pico_command = f'pico2wave -l {language} -w speech.wav "{text}"'
    
    # Execute the pico2wave command
    os.system(pico_command)
    
    # Play the generated speech using aplay
    os.system('aplay speech.wav')
    
    # Clean up the temporary speech file
    os.remove('speech.wav') 
#----------------------------------------
    
def convert_string_to_time(time_str):
    # Parse the time string using strptime
     #time_obj=datetime.now()
    time_str = time_str.replace(".", "")
    try:
      time_obj = datetime.strptime(time_str, "%I:%M %p")
      time_value = time_obj.time()
      time_value=time_value.strftime("%H:%M:%S")
      return time_value
    except: 
      print("Could not parse") #ignore

    try:
      time_obj = datetime.strptime(time_str, "%I%M %p")
      time_value = time_obj.time()
      time_value=time_value.strftime("%H:%M:%S")
      return time_value
    except: return None
      
#-----------------------------------------

def SetReminder(Description,time):

  #check if user gave correct time format
  convert_str_time=convert_string_to_time(time)
  if not convert_str_time:
    say("Did not provide proper time")
    return
  #connect to database
  conn=sqlite3.connect('reminder.db')
  c=conn.cursor() #connect to cursor
  time_format=convert_str_time
  try: 
    c.execute("INSERT INTO VoiceReminder VALUES (?, ?)", (Description, time_format))
    say("Reminder Stored Successfully at %s"%(time_format))
  except:
    say("Could not Store Reminder")
  finally:
    conn.commit()
    conn.close()
    return
#---------------------------------------------

def GetReminder(time):
  conn=sqlite3.connect('reminder.db')
  c=conn.cursor() #connect to cursor
  desc_msg = c.execute("SELECT desc FROM VoiceReminder WHERE time=?", (time,)).fetchone()
  #if desc_msg != None
  if desc_msg:
   desc_msg="Reminder "+desc_msg[0]
   print(desc_msg)
   say(desc_msg)
  conn.close()
  return
  
#--------------------------------------------------
def DeleteReminder(time):
 
  #if user stays silent
  if(time=="none"):
    say("Did not get any instructions")
    return
  #check if user gave correct time format
  convert_str_time=convert_string_to_time(time)
  if convert_str_time==None:
    say("Did not provide proper time")
    return
  #storing correct time format for further operations
  time_format=convert_str_time 
 
  #Make Connection to the Database
  conn=sqlite3.connect('reminder.db')
  c=conn.cursor() 
  #check if the said time exists in database or not
  check_time=c.execute("SELECT time FROM VoiceReminder WHERE time=?", (time_format,)).fetchone()
  print(check_time)
  if not check_time: #alternate statement: if check_time==None 
    say("No Reminder Exist at %s"%(time))
    conn.close()
    return
  #if time exists :)
  c.execute("DELETE FROM VoiceReminder WHERE time=?",(time_format,))
  say("Successfully Deleted Reminder")
  conn.commit()  # Don't forget to commit the changes
  conn.close()
  return

#----------------------------------------------------------------------------





#------Code for Print values in DataBase (For Testing Purpose only)---------
#DeleteReminder("10:11 am")
# SetReminder("Fitness first","8:00 am")
# #GetReminder("10:00:00")
# conn=sqlite3.connect('reminder.db')
# c=conn.cursor() #connect to cursor
# c.execute("SELECT rowid, * FROM VoiceReminder")
# print(c.fetchall())

# say("hello candy")