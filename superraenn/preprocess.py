import sys
sys.path.append("..")

from argparse import ArgumentParser


def main():
	parser = ArgumentParser()
	parser.add_argument('data dir', help='Directory of LC files')
	parser.add_argument('metatable', type=int, help='Metatable containing each object, redshift, peak time guess, mwebv, zeropoint, object type, and limiting magnitude')
	args = parser.parse_args()

if __name__ == '__main__':
    main()