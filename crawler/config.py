import os

## CONFIGURATION VARIABLES FOR THE GLOBAL SYSTEM
class GlobalConfig:
  def __init__(self, argv):
    self.vars = argv
    # PATHS
    self.sysdir = argv["sysdir"]

  def log(self, dbhome, message):
    os.system("t=$(date); echo $t "+message+" >> "+dbhome+"/dbinfo/events.txt")

## FUNCTION TO ADD ZEROS TO THE NUMBER OF A BATCH
def formatBatch(b):
  return str(b).zfill(5)

## LOAD MAIN CONFIGURATION VARIABLES
#print os.getcwd()
vars = dict([line.replace('\n','').split("=") for line in open('config.txt', 'r').readlines() if not line.startswith("#") ])
cfg = GlobalConfig(vars)


