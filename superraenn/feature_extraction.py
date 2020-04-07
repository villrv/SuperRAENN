import numpy as np
from lc import LightCurve

def read_in_LC_files(input_files,obj_names, style='SNANA'):
	"""
	Read in LC files and convert to LC object

	Parameters
	----------
	input_files : list
		List of LC file names, to be read in.
	obj_names : list
		List of SNe names, should be same length as input_files
	style : string
		Style of LC files. Assumes SNANA

	Returns
	-------
	lcs : list
		list of Light Curve objects

	Examples
	--------
	"""
	LC_list = []
	if style == 'SNANA':
		for i,input_file in enumerate(input_files):
			t,f,filts,err = np.genfromtxt(input_file,\
							usecols=(1,4,2,5),skip_header=18,
							skip_footer=1,unpack=True,dtype=str)
			t = np.asarray(t,dtype=float)			
			f = np.asarray(f,dtype=float)
			err = np.asarray(err,dtype=float)

			sn_name = obj_names[i]
			new_LC = LightCurve(sn_name,t,f,err,filts)
			LC_list.append(new_LC)
	else:
		raise ValueError('Sorry, you need to specify a data style.')
	return LC_list


def feat_peaks(input_lcs):
	"""
	Extract peak magnitudes from GP LCs

	Parameters
	----------
	input_lcs : list
		List of LC objects

	Returns
	-------
	peaks : list
		Peaks from each LC filter

	Examples
	--------
	"""
	peaks = []
	for input_lc in input_lcs:
		peaks.append(np.nanmin(input_lc.dense_lc[:,:,0],axis=0))
	return peaks

def feat_rise_and_decline(input_lcs, gps, gp_mags_list,n_mag,nfilts=4):

	t_falls_all = []
	t_rises_all = []

	for i,input_lc in enumerate(input_lcs):
		gp = gps[i]
		gp_mags = gp_mags_list[i]
		t_falls = []
		t_rises = []
		for j in np.arange(nfilts):
			new_times = np.linspace(-100,100,1000)
			x_stacked = np.asarray([new_times,[j]*1000]).T
			pred,var = gp.predict(gp_mags,x_stacked)

			max_ind = np.nanargmin(pred)
			max_mag = pred[max_ind]
			max_t = new_times[max_ind]
			trise = np.where((new_times<max_t) & (pred>(max_mag+n_mag)))
			tfall = np.where((new_times>max_t) & (pred>(max_mag+n_mag)))
			if len(trise[0]) == 0:
				trise = np.max(new_times) - max_t
			else:
				trise = max_t - new_times[trise][-1]
			if len(tfall[0]) == 0:
				tfall = max_t - np.min(new_times)
			else:
				tfall = new_times[tfall][0] - max_t

			t_falls.append(tfall)
			t_rises.append(trise)
		t_falls_all.append(t_falls)
		t_rises_all.append(t_rises)
	return t_rises_all, t_falls_all

def feat_slope(input_lcs, gps, gp_mags_list, t_min_lim=10, \
				t_max_lim=30, nfilts=4):
	slopes_all = []
	for i,input_lc in enumerate(input_lcs):
		gp = gps[i]
		gp_mags = gp_mags_list[i]
		slopes = []
		for j in np.arange(nfilts):
			new_times = np.linspace(-100,100,1000)
			x_stacked = np.asarray([new_times,[j]*1000]).T
			pred,var = gp.predict(gp_mags,x_stacked)
			max_ind = np.nanargmin(pred)
			max_mag = pred[max_ind]
			max_t = new_times[max_ind]
			new_times = new_times - max_t
			lc_grad = np.gradient(pred,new_times)
			gindmean = np.where((new_times>t_min_lim) & (new_times<t_max_lim))
			slopes.append(np.nanmedian(lc_grad[gindmean]))
		slopes_all.append(slopes)
	return slopes_all


def feat_int(input_lcs, gps, gp_mags_list, nfilts=4):
	ints_all = []
	for i,input_lc in enumerate(input_lcs):
		gp = gps[i]
		gp_mags = gp_mags_list[i]
		ints = []
		for j in np.arange(nfilts):
			new_times = np.linspace(-100,100,1000)
			x_stacked = np.asarray([new_times,[j]*1000]).T
			pred,var = gp.predict(gp_mags,x_stacked)
			ints.append(np.trapz(pred))

		ints_all.append(ints)
	return ints_all
