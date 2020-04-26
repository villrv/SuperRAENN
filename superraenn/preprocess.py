from argparse import ArgumentParser
import numpy as np
from .feature_extraction import read_in_LC_files
import logging
import datetime

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Default fitting parameters
DEFAULT_ZPT = 27.5
DEFAULT_LIM_MAG = 25.0

def read_in_meta_table(metatable):
	"""
	Read in the metatable file
	Parameters
	----------
	metatable : str
		Name of metatable file
	Returns
	-------
	obj : numpy.ndarray
		Array of object IDs (strings)
	redshift : numpy.ndarray
		Array of redshifts
	obj_type : numpy.ndarray
		Array of SN spectroscopic types
	my_peak : numpy.ndarray
		Array of best-guess peak times
	ebv : numpy.ndarray
		Array of MW ebv values

	Todo
	----------
	Make metatable more flexible
	"""
	obj,redshift,obj_type,my_peak, ebv = np.loadtxt(metatable,unpack=True,\
										dtype=str, delimiter=' ')
	redshift = np.asarray(redshift,dtype=float)
	my_peak = np.asarray(my_peak,dtype=float)
	ebv = np.asarray(ebv,dtype=float)

	return obj,redshift,obj_type,my_peak, ebv

def save_lcs(lc_list,output_dir):
	"""
	Save light curves as a lightcurve object
	Parameters
	----------
	lc_list : list
		list of light curve files
	output_dir : Output directory of light curve file

	Todo:
	----------
	- Add option for LC file name
	"""
	now = datetime.datetime.now()
	date = str(now.strftime("%Y-%m-%d"))
	file_name = 'lcs_'+date+'.npz'
	output_file = output_dir+file_name
	np.savez(output_file,lcs = lc_list)
	logging.info(f'Saved to {output_file}')

def main():
	"""
	Preprocess the LC files
	"""

	# Create argument parser
	parser = ArgumentParser()
	parser.add_argument('datadir', type=str, help='Directory of LC files')
	parser.add_argument('metatable', type=str, help='Metatable containing each object, redshift, peak time guess, mwebv, object type')
	parser.add_argument('--zpt', type=float, default=DEFAULT_ZPT, help='Zero point of LCs')
	parser.add_argument('--lm', type=float, default=DEFAULT_LIM_MAG, help='Survey limiting magnitude')
	parser.add_argument('--outdir', type=str, default='./', help='Path in which to save the LC data (single file)')
	args = parser.parse_args()

	objs, redshifts, obj_types, peaks, ebvs = \
						read_in_meta_table(args.metatable)
	
	# Grab all the LC files in the input directory
	file_names = []
	for obj in objs:
		file_name = args.datadir + 'PS1_PS1MD_'+obj+'.snana.dat'
		file_names.append(file_name)

	# Create a list of LC objects from the data files
	lc_list = read_in_LC_files(file_names,objs)

	# This needs to be redone when retrained
	# TODO: Need to change this whenever you retrain...
	filt_dict = {'g':0,'r':1,'i':2,'z':3}
	wvs = np.asarray([5460,6800,7450,8700])

	#Update the LC objects with info from the metatable
	my_lcs = []
	for i,my_lc in enumerate(lc_list):
		my_lc.add_LC_info(zpt = args.zpt, mwebv = ebvs[i], \
							redshift = redshifts[i], lim_mag = args.lm, \
							obj_type = obj_types[i])
		my_lc.get_abs_mags()
		my_lc.sort_lc()
		pmjd = my_lc.find_peak(peaks[i])
		my_lc.shift_lc(pmjd)
		my_lc.correct_time_dilation()
		my_lc.filter_names_to_numbers(filt_dict)
		my_lc.correct_extinction(wvs)
		my_lc.cut_lc()
		my_lc.make_dense_LC(4)
		my_lcs.append(my_lc)
	save_lcs(my_lcs,args.outdir)

if __name__ == '__main__':
    main()