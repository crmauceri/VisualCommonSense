import os, sys

# ######################################
# DATABASE DIRS MANAGEMENT
# ######################################
class DBDirectories:
  def __init__(self, home, sys, category):
    self.sys = sys
    self.home = home
    subdir = {"txt":"/texts/", "img":"/images/", "inf":"/dbinfo/"}
    batch = {"txt":True, "img":True, "inf":False}
    for key in subdir:
      val = self.initDir(home,subdir[key],category,batch[key])
      setattr(self, key, val)

  def initDir(self, path, subdir, category, batch):
    targetDir = path + subdir
    if not os.path.exists(targetDir):
      os.mkdir(targetDir)
    targetDir = targetDir + category + '/'
    if not os.path.exists(targetDir):
      os.mkdir(targetDir)
      if batch:
        os.mkdir(targetDir+'/00001/')
    return targetDir

  def uploadCurrentDirAndGetNext(self, maxFiles, queries):
    dirNum = max(map(int, os.listdir(self.txt)))
    dirNumName = str(dirNum).zfill(5) + '/'
    if len( os.listdir(self.txt + dirNumName) ) >= maxFiles:
      allObjects = reduce(lambda x,y:x+'_'+y, queries)
      #os.system('sh '+self.sys+'/shell/upload.sh '+self.sys+' '+dirNumName.replace('/','')+' '+allObjects+' & ')
      #print '\nUploading batch',dirNumName,'in background...\n'
      with open(self.inf+'downloads.txt', 'a') as fingerprint:
        fingerprint.write(dirNumName+' '+allObjects+'\n')
      dirNum += 1
      dirNumName = str(dirNum).zfill(5) + '/'
      os.mkdir(self.txt+dirNumName)
      os.mkdir(self.img+dirNumName)
    return dirNumName

