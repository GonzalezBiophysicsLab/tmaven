#!/usr/bin/env python

import os
import argparse

def file_path(string):
	if os.path.exists(string):
		return string
	else:
		err = "\n\nError: file %s does not exist\n\n"%(string)
		raise Exception(err)

def main():
	parser = argparse.ArgumentParser(description="(t)ime-series (M)odeling, (A)nalysis, and (V)isualization (EN)vironment")
	
	parser.add_argument('--safe',action='store_true',default=False,help='Open in safe-mode')
	parser.add_argument('--log',action='store_true',default=False,help='Send log messages to stdout')
	
	parser.add_argument('--script',type=file_path,help='Specify a script to run on startup')

	args = parser.parse_args()
	input_args = []
	if args.safe:
		input_args.append('--safe_mode')
	if args.log:
		input_args.append('--log_stdout')
	
	if not args.script is None:
		input_args.append('--startup=%s'%(args.script))
	
	from tmaven import run_app
	run_app(input_args)

if __name__ == '__main__':
	main()


