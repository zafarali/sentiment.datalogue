import os, sys

def add_module_paths():
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	print 'Added root as path'
