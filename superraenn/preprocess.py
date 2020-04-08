import sys
sys.path.append("..")

#For testing, use python3 preprocess.py /Volumes/My\ Passport/ps1_sne/all_lcs/output_lc/ /Users/ashley/Dropbox/Research/ml_ps1_full/nn_clean/superraenn_tester/meta_table_040720.dat

from argparse import ArgumentParser
import numpy as np
from feature_extraction import read_in_LC_files
import logging
import datetime

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
# Default fitting parameters
DEFAULT_ZPT = 27.5
DEFAULT_LIM_MAG = 25.0

def read_in_meta_table(metatable):
	obj,redshift,obj_type,my_peak, ebv = np.loadtxt(metatable,unpack=True,\
										dtype=str, delimiter=' ')
	redshift = np.asarray(redshift,dtype=float)
	my_peak = np.asarray(my_peak,dtype=float)
	ebv = np.asarray(ebv,dtype=float)

	return obj,redshift,obj_type,my_peak, ebv

def save_lcs(lc_list,output_dir):
	now = datetime.datetime.now()
	date = str(now.strftime("%Y-%m-%d"))
	file_name = 'lcs_'+date+'.npz'
	output_file = output_dir+file_name
	np.savez(output_file,lcs = lc_list)
	logging.info(f'Saved to {output_file}')

def main():
	parser = ArgumentParser()
	parser.add_argument('datadir', type=str, help='Directory of LC files')
	parser.add_argument('metatable', type=str, help='Metatable containing each object, redshift, peak time guess, mwebv, object type')
	parser.add_argument('--zpt', type=float, default=DEFAULT_ZPT, help='Zero point of LCs')
	parser.add_argument('--lm', type=float, default=DEFAULT_LIM_MAG, help='Survey limiting magnitude')
	parser.add_argument('--outdir', type=str, default='.', help='Path in which to save the LC data (single file)')
	args = parser.parse_args()

	objs, redshifts, obj_types, peaks, ebvs = \
						read_in_meta_table(args.metatable)
	
	file_names = []
	for obj in objs:
		file_name = args.datadir + 'PS1_PS1MD_'+obj+'.snana.dat'
		file_names.append(file_name)

	lc_list = read_in_LC_files(file_names,objs)
	filt_dict = {'g':0,'r':1,'i':2,'z':3}


	my_lcs = []
	lc_list=lc_list[0:2]
	for i,my_lc in enumerate(lc_list):
		my_lc.add_LC_info(zpt = args.zpt, mwebv = ebvs[i], \
							redshift = redshifts[i], lim_mag = args.lm, \
							obj_type = obj_types[i])

		my_lc.get_abs_mags()
		my_lc.sort_lc()
		pmjd = my_lc.find_peak(peaks[i])
		my_lc.shift_lc(pmjd)
		my_lc.cut_lc(100)
		my_lc.filter_names_to_numbers(filt_dict)
		my_lc.make_dense_LC(4)
		my_lcs.append(my_lc)

	save_lcs(my_lcs,args.outdir)



if __name__ == '__main__':
    main()