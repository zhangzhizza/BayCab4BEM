import numpy as np;

class EtaKernel(object):

	def getValue(self, row_i, row_j, beta_x, beta_t, lambda_eta):
		"""
		The kernel function for eta (\Sigma_eta). Function B.3 of the Adrian Chong's PhD thesis (2017)

		Args:
			row_i: np.ndarray 
				1-D array, row i of the combined data
			row_j: np.ndarray
				1-D array, row j of the combined data
			beta_x: np.ndarray
				1-D array, betas for x correlation.
			beta_t: np.ndarray
				1-D array, betas for t correlation.
			lambda_eta: float
				The lambda value. 

		Ret:
			float.
		"""
		row_i_x_part = row_i[0: beta_x.shape[0]];
		row_i_t_part = row_i[beta_x.shape[0]:];
		row_j_x_part = row_j[0: beta_x.shape[0]];
		row_j_t_part = row_j[beta_x.shape[0]:];

		x_diff_abs_square = (abs(row_i_x_part - row_j_x_part))**2.0;
		t_diff_abs_square = (abs(row_i_t_part - row_j_t_part))**2.0;

		x_diff_abs_square_multi_beta = np.multiply(x_diff_abs_square, beta_x);
		t_diff_abs_square_multi_beta = np.multiply(t_diff_abs_square, beta_t);

		sum_together = -sum(x_diff_abs_square_multi_beta) - sum(t_diff_abs_square_multi_beta);

		ret = np.exp(sum_together)/lambda_eta;

		return ret;

def deltaKernel(row_i, row_j, beta_x, lambda_delta):
	"""
	The kernel function for delta (\Sigma_delta). Function B.4 fo the Adrian Chong's PhD thesis (2017)

	Args:
		row_i: np.ndarray 
			1-D array, row i of the field data
		row_j: np.ndarray
			1-D array, row j of the field data
		beta_x: np.ndarray
			1-D array, betas for x correlation.
		lambda_delta: float
			The lambda value. 

	Ret:
		float.
	"""
	x_diff_abs_square = (abs(row_i- row_j))**2.0;

	x_diff_abs_square_multi_beta = np.multiply(x_diff_abs_square, beta_x);

	sum_together = -sum(x_diff_abs_square_multi_beta);

	ret = np.exp(sum_together)/lambda_delta;

	return ret;