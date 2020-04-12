import datetime
import logging
from argparse import ArgumentParser
import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


now = datetime.datetime.now()
date = str(now.strftime("%Y-%m-%d"))

def KDE_resample(x,y,N):
	uys = np.unique(y)
	newX = np.zeros((int(N*len(uys)),np.size(x,axis=1)))
	newy = np.zeros((int(N*len(uys)),))
	for i,uy in enumerate(uys):
		gind = np.where(y==uy)
		newX[i*N:i*N+len(gind[0]),:] = x[gind[0],:]
		newy[i*N:(i+1)*N] = uy
		cx = x[gind[0],:]
		kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(cx)
		newX[i*N+len(gind[0]):(i+1)*N] = kde.sample(n_samples=N-len(gind[0]))
	return newX,newy

def prep_data_for_training(featurefile, metatable, whiten=True):
	feat_data = np.load(featurefile, allow_pickle=True)
	ids = feat_data['ids']
	features = feat_data['features']
	feat_names = feat_data['feat_names']
	metadata = np.loadtxt(metatable, dtype=str, usecols=(0,2))
	sn_dict = {'SLSN':0, 'SNII':1, 'SNIIn':2, 'SNIa':3, 'SNIbc':4}

	X = []
	y = []
	final_sn_names = []
	for sn_name,sn_type in metadata:
		gind = np.where(sn_name==ids)
		if 'SN' not in sn_type:
			continue
		else:
			sn_num = sn_dict[sn_type]

		if not np.isfinite(features[gind][0]).all():
			continue

		if X == []:
			X = features[gind][0]
			y = sn_num
		else:
			X = np.vstack((X,features[gind][0]))
			y = np.append(y,sn_num)
		final_sn_names.append(sn_name)
	
	if whiten:
		means = np.mean(X,axis=0)
		stds = np.std(X,axis=0)
		X = preprocessing.scale(X)

	return X,y,final_sn_names, means,stds, feat_names


def main():
	parser = ArgumentParser()
	parser.add_argument('featurefile', type=str, help='Feature file')
	parser.add_argument('--metatable', type=str, default='', help='Get training set labels')
	parser.add_argument('--outdir', type=str, default='./', help='Path in which to save the LC data (single file)')
	parser.add_argument('--train', type = bool, default = True, help = '...')
	parser.add_argument('--add-random', dest='add_random',type = bool, default = False, help = '...')
	parser.add_argument('--calc-importance', dest='calc_importance',type = bool, default = False, help = '...')

	args = parser.parse_args()

	X,y,names, means,stds, feature_names = prep_data_for_training(args.featurefile,args.metatable)
	names = np.asarray(names,dtype=str)
	X = X[:,0:10]
	feature_names = feature_names[0:10]
	if args.add_random:
		feature_names = np.append(feature_names,'random')

	loo = LeaveOneOut()
	import matplotlib.pyplot as plt
	for train_index, test_index in loo.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		if y_test[0]!=4:
			continue
		
		X_res, y_res = KDE_resample(X_train, y_train,400)
		new_ind = np.arange(len(y_res),dtype=int)
		np.random.shuffle(new_ind)
		X_res = X_res[new_ind]
		y_res = y_res[new_ind]
		X_res2, y_res2 = KDE_resample(X_train, y_train,400)
		X_res2 = X_res2[:-40,:]
		y_res2 = y_res2[:-40]

		if args.add_random:
			X_res = np.vstack((X_res.T,np.random.randn(len(X_res)))).T
			X_res2 = np.vstack((X_res2.T,np.random.randn(len(X_res2)))).T
			X_test = np.vstack((X_test.T,np.random.randn(len(X_test)))).T


		clf = RandomForestClassifier(n_estimators=300, max_depth=None,
						random_state=42, criterion='gini',class_weight='balanced',
									 max_features=None,oob_score=False)
		clf.fit(X_res,y_res)
		print(clf.predict_proba(X_test),y_test,names[test_index])

		if args.calc_importance:
			result = permutation_importance(clf, X_res2, y_res2, 
								n_repeats=10,
                                random_state=42, n_jobs=2)

			indices = result.importances_mean.argsort()[::-1]
			print(result.importances_mean)
			feature_names = np.asarray(feature_names,dtype=str)
			importances = result.importances_mean
			std = result.importances_std

			# Print the feature ranking
			print("Feature ranking:")

			for f in range(X_res.shape[1]):
			    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

			plt.ylabel("Feature importances")
			plt.bar(range(X_res.shape[1]), importances[indices],
			       color="grey", yerr=std[indices], align="center")
			plt.xticks(np.arange(len(importances))+0.5, feature_names[indices],
						rotation=45,ha='right')
			plt.show()


if __name__ == '__main__':
	main()