##############################################################################################################################################
# AUTHOR: KUNAL PALIWAL
# EMAIL ID: kupaliwa@syr.edu
# COURSE: ARTIFICAL NEURAL NETWORKS 
# This file is responsible for parsing whatsapp data and building a csv file
##############################################################################################################################################
import numpy as np
np.random.seed(0)
import sys
import pandas as pd
import os
from os import path
import csv
import codecs

class Parser():
    # --------------------------------< Initializing parameters (Constructor) >-------------------------------    
    def __init__(self, inputFileName,outputFileName):
        self.inputFileName = inputFileName
        self.outputFileName = outputFileName
        self.raw_messages = []
        self.speakerlist = []
        self.messagelist = []
        self.paragraphList = []

        self.datelist = []
        self.timelist = []

# --------------------------------< Read Whatsapp chatlog >-------------------------------            
    def open_file(self):
        arq = codecs.open(self.inputFileName, "r", "utf-8-sig")
        content = arq.read()
        arq.close()
        lines = content.split("\n")
        lines = [l for l in lines if len(l) > 4]
        for l in lines:
            self.raw_messages.append(l.encode("utf-8"))

# --------------------------------< Generate output >-------------------------------    
    def build_csv(self, end=0):
        if end == 0:
            end = len(self.messagelist)
        writer = csv.writer(open(self.outputFileName, 'w', encoding='utf-8'))
        writer.writerow(["SentenceNo","SequenceNo","Date","Time","Speaker","Text"])
        for i in range(len(self.messagelist[:end])):
            writer.writerow([i,self.paragraphList[i],self.datelist[i], self.timelist[i],self.speakerlist[i], self.messagelist[i]])
    
# --------------------------------< check if data is defined or not >-------------------------------    
    def checkdef(self,data):
        try: data
        except: data = " "

# --------------------------------< Validate date >-------------------------------    
    def valid_date(self,date_str):
        valid = True
        separator="/"
        try:
            year, month, day = map(int, date_str.split(separator))
        except ValueError:
            valid = False
        return valid


# --------------------------------< Feed into list >-------------------------------    
    def feed_lists(self):
        lineNo = 0
        seqNo = 0
        for l in self.raw_messages:
            l = l.rstrip()
            msg_date, sep, msg = l.decode().partition(": ")
            raw_date = ""
            sep = ""
            time = ""
            #Date and time has a , separator
            raw_date, sep, time = msg_date.partition(", ")
            speaker = ""
            sep = ""
            message = ""
            speaker, sep, message = msg.partition(": ")
            #speaker = speaker.encode('utf-8')
            lineNo += 1
            # A proper whatsapp conversation with date, time, speaker, text
            if message:
                self.datelist.append(raw_date)
                self.timelist.append(time)
                self.speakerlist.append(speaker)
                self.messagelist.append(message)
                # store the previous speaker so that you can use it to print when there is only a line
                prevSender = speaker
                prevRawDate = raw_date
                prevTime = time
                seqNo +=1
            # A message. date, time, message
            elif ((speaker != "") & (self.valid_date(raw_date))):
                self.datelist.append(raw_date)
                self.timelist.append(time)
                self.speakerlist.append('MESSAGE')
                self.messagelist.append(speaker)
                # store the previous speaker so that you can use it to print when there is only a line
                prevSender = 'MESSAGE'
                prevRawDate = raw_date              
                prevTime = time
                seqNo +=1
    
            else:
                try: prevRawDate
                except: prevRawDate = " "
                try: prevTime
                except: prevTime = " "
                try: prevSender
                except: prevSender = " "
                self.checkdef(prevRawDate)
                self.checkdef(prevTime)
                self.checkdef(prevSender)
                self.checkdef(l)                
                self.datelist.append(prevRawDate)
                self.timelist.append(prevTime)
                self.speakerlist.append(prevSender)
                self.messagelist.append(l)
            self.paragraphList.append(seqNo)

# --------------------------------< Wrapper for extracting data >-------------------------------    
def config():    
    c_parser = Parser("data.txt", "output.csv")
    c_parser.open_file()
    c_parser.feed_lists()
    c_parser.build_csv()

if __name__ == "__main__":
    print('testing')
    config();
    # parse_whatsapp()    