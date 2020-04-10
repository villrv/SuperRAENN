import numpy as np
from .lc import LightCurve
from .raenn import prep_input, get_encoder, get_decoder, get_decodings
from argparse import ArgumentParser
from keras.models import model_from_json, Model
from keras.layers import Input
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


def feat_from_raenn(data_file, model_base = None, \
					prep_file = None, generate = True):
	if generate:
		sequence, outseq, ids, maxlen, nfilts = prep_input(data_file,load=True, prep_file=prep_file)
		model_file = model_base + '.json'
		model_weight_file = model_base+'.h5'
		with open(model_file, 'r') as f:
			model = model_from_json(f.read())
		model.load_weights(model_weight_file)

		encodingN = model.layers[2].output_shape[1]
		encoded_input = Input(shape=(None,(encodingN+2)))
		original_input = Input(shape=(None,nfilts*2+1))
		decoder_layer2 = model.layers[-2]
		decoder_layer3 = model.layers[-1]
		merged = model.layers[-3]
		repeater = model.layers[-4]
		encoded = model.layers[2]
		encoded1 = model.layers[1]

		#test_model(sequence_test,model,lm, maxlen, plot=True)
		encoder = Model(input=original_input, output=encoded(encoded1(original_input)))
		encodings = np.zeros((len(ids),encodingN))
		for i in np.arange(len(ids)):
			inseq = np.reshape(sequence[i,:,:],(1,maxlen,nfilts*2+1))
			my_encoding = encoder.predict(inseq)
			encodings[i,:] = my_encoding
			encoder.reset_states() 
		return encodings

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


def main():
	parser = ArgumentParser()
	parser.add_argument('lcfile', type=str, help='Light curve file')
	parser.add_argument('--outdir', type=str, default='.', help='Path in which to save the LC data (single file)')
	parser.add_argument('--plot', type=bool, default = False, help='Plot LCs')
	parser.add_argument('--model-base', type=str, dest='model_base', default = '', help='...')
	parser.add_argument('--feat-encode', type=bool, dest='feat_encode', default=True, help='...')
	parser.add_argument('--prep-file', type=str, dest='prep_file', default='', help='...')

	args = parser.parse_args()

	if args.feat_encode:
		print(feat_from_raenn(args.lcfile,model_base = args.model_base, prep_file=args.prep_file))

	

if __name__ == '__main__':
	main()
