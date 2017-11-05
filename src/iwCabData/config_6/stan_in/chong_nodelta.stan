data {
	int <lower=0> n; // number of observations
	int <lower=0> m; // number of simulations
	int <lower=0> N; // N=n+m
	int <lower=0> p; // number of input factors
	int <lower=0> q; // number of calibration parameters
	matrix [n, p] xf; //
	matrix [m, p] xc; //
	matrix [m, q] t; // calibration parameters
	vector [N] z;
}
transformed data {
	vector [N] mu_z;
	for (i in 1:N){
		mu_z[i] = 0;
	}
}
parameters {
	row_vector <lower=0, upper=1> [q] theta; // calibration parameters
	row_vector <lower=0, upper=1> [p+q] rho_eta;
	row_vector <lower=0, upper=1> [p] rho_delta;
	real <lower=0> lambda_eta; // precision parameter for eta
	real <lower=0> lambda_delta; // precision parameter for bias
	real <lower=0> lambda_e; // precision parameter for observation error
}
transformed parameters {
	// declare variables
	row_vector [p+q] beta_eta;
	row_vector [p] beta_delta;
	beta_eta = -4.0*(log(rho_eta));
	beta_delta = -4.0*(log(rho_delta));
}
model {
	// declare variables
	matrix [N,(p+q)] inputs;
	matrix [N,N] sigma_eta;
	matrix [n,n] sigma_delta;
	matrix [N,N] sigma_z;
	matrix [n,n] sigma_y;
	matrix [N,N] L; // cholesky decomposition of covariance matrix
	row_vector [p] temp_delta;
	row_vector [p+q] temp_eta;
	// set values of inputs which would be used to compute sigma_eta
	inputs [1:n,1:p] = xf;
	inputs [(n+1):N, 1:p] = xc;
	inputs [1:n, (p+1):(p+q)] = rep_matrix(theta,n);
	inputs [(n+1):N, (p+1):(p+q)] = t;
	// diagonal elements of sigma_eta
	sigma_eta = diag_matrix(rep_vector((1/lambda_eta),N));
	// off diagonal elements of sigma_eta
	for(i in 1:(N-1)) {
		for (j in (i+1):N) {
			temp_eta = inputs[i,1:(p+q)] - inputs[j,1:(p+q)];
			sigma_eta[i,j] = beta_eta .* temp_eta*temp_eta';
			sigma_eta[i,j] = exp(-sigma_eta[i,j]) / lambda_eta;
			sigma_eta[j,i] = sigma_eta[i,j];
		}
	}
	// diagonal elements of sigma_delta
	// sigma_delta = diag_matrix(rep_vector((1/ lambda_delta),n));
	// off diagonal elements of sigma_delta
	//for(i in 1:(n-1)) {
	//	for (j in (i+1):n) {
	//		temp_delta = xf[i,1:p] - xf[j,1:p];
	//		sigma_delta[i,j] = beta_delta .* temp_delta * temp_delta';
	//		sigma_delta[i,j] = exp(-sigma_delta[i,j]) / lambda_delta;
	//		sigma_delta[j,i] = sigma_delta [i,j];
	//	}
	//}
	// diagonal elements of sigma e with off diagonal elements left as 0
	sigma_y = diag_matrix (rep_vector((1.0/lambda_e),n));
	// computation of covariance matrix sigma_z
	sigma_z = sigma_eta;
	sigma_z[1:n,1:n] = sigma_eta[1:n,1:n] + sigma_y;
	// Priors
	for (i in 1:(p+q)){
		rho_eta[i] ~ beta(1,0.5);
	}
	for (j in 1:p){
		rho_delta[j] ~ beta(1,0.4);
	}
	for (k in 1:q){
		theta[k] ~ uniform(0.0, 1.0);
	}
	lambda_eta ~ gamma(10, 10); // gamma(shape, rate)
	lambda_delta ~ gamma(10, 0.3); // gamma(shape, rate)
	lambda_e ~ gamma(10, 0.03); // gamma(shape, rate)
	// cholesky decomposition of covariance matrix
	L = cholesky_decompose(sigma_z);
	z ~ multi_normal_cholesky(mu_z,L);
}