import numpy as np

def helloworld(input):
	my_message = 'hi '+input
	"""
	Compute the refraction angle using Snell's Law.

	Parameters
	----------
	theta_inc : float
		Incident angle in radians.
	n1, n2 : float
		The refractive index of medium of origin and destination medium.

	Returns
	-------
	theta : float
		refraction angle

	Examples
	--------
	A ray enters an air--water boundary at pi/4 radians (45 degrees).
	Compute exit angle.

	>>> snell(np.pi/4, 1.00, 1.33)
	0.5605584137424605
	"""
	return my_message