import sys
sys.path.append("..")

from feature_extraction import read_in_LC_files
from argparse import ArgumentParser


def main():
	parser = ArgumentParser()
	parser.add_argument('data file', help='Filename of saved data table')
	parser.add_argument('--random-state', type=int, help='Seed for the random number generator (for reproducibility).')
	args = parser.parse_args()

if __name__ == '__main__':
    main()
