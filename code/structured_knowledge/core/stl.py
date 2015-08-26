def getFileOfType(dir, prefix, type):
	import os
	real_dir = os.path.abspath(dir)
#	os.chdir(dir)
	ret = []
	for filepath in os.listdir(real_dir):
		if filepath.endswith(prefix+"."+str(type)):
			ret.append(filepath)
	return ret
