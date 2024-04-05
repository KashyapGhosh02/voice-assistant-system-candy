import sqlite3
from datetime import datetime
conn=sqlite3.connect('reminder.db')
c=conn.cursor()
c.execute("""CREATE TABLE VoiceReminder(
          desc text,
          time text
)""")