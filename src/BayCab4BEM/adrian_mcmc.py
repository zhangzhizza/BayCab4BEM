import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano
import matplotlib.pyplot as plt
from pymc3.step_methods.metropolis import Metropolis


D_COMP = np.genfromtxt('cal_example_com_withoutSummer.csv', delimiter = ',')
D_FIELD = np.genfromtxt('cal_example_field_withoutSummer.csv', delimiter = ',')


y = D_FIELD[:,0]
xf = D_FIELD[:,1:]
(n,p) = xf.shape
eta = D_COMP[:,0]
xc = D_COMP[:,1:(p+1)]
tc = D_COMP[:,(p+1):]
(m,q) = tc.shape

eta_mu = np.nanmean(eta)
eta_sd = np.nanstd(eta)
y = (y - eta_mu) / eta_sd
eta = (eta - eta_mu) /eta_sd
z = np.concatenate((y,eta), axis=0)

x = np.concatenate((xf,xc), axis=0)
x = (x - x.min(axis = 0)) / x.ptp(axis = 0)
xf = x[0:n,:]
xc = x[n:,:]
t = (tc - tc.min(axis = 0)) / tc.ptp(axis = 0)

x_shared = theano.shared(x)
z_shared = theano.shared(z)

with pm.Model() as model:
	# priors on the calibration parameters
	theta= pm.Uniform('tf', lower=0, upper=1, shape=q)

	# priors on the covariance function hyperparameters
	rho_eta = pm.Beta('rho_eta', alpha=1, beta=0.5, shape=(p+q))
	beta_eta = -4 * (tt.log(rho_eta))
	rho_delta = pm.Beta('rho_delta', alpha=1, beta=0.4, shape=p)
	beta_delta = -4 * (tt.log(rho_delta))
	lambda_eta = pm.Gamma('lambda_eta',alpha=5., beta=5.)
	lambda_delta = pm.Gamma('lambda_delta', alpha=1., beta=0.00001)
	lambda_e = pm.Gamma('lambda_e', alpha=1., beta=0.00001)
	#lambda_en = pm.Gamma('lambda_en', alpha=1., beta=0.00001)
	data_right = tt.concatenate([tt.tile(theta,(n,1)),t], axis=0)
	data = tt.concatenate([x,data_right], axis=1)


	l_eta = tt.sqrt(0.5 * beta_eta)
	l_delta = tt.sqrt(0.5 * beta_delta)

	#X = tt.tile(tf,(n,1))
	sigma_eta_op = (1/lambda_eta) * pm.gp.cov.ExpQuad(input_dim=(p+q), lengthscales=l_eta)
	sigma_delta_op = (1/lambda_delta) * pm.gp.cov.ExpQuad(input_dim=p, lengthscales=l_delta)
	sigma_e = (1/lambda_e) * tt.eye(n)
	#sigma_en = (1/lambda_en) * tt.eye(m)
	
	sigma_eta = sigma_eta_op(data)
	sigma_delta = sigma_delta_op(xf)
	sigma_z = tt.set_subtensor(sigma_eta[0:n,0:n],sigma_eta[0:n,0:n]+sigma_delta+sigma_e)
	#new_new_cov = tt.set_subtensor(new_cov[n:,n:], new_cov[n:,n:]+sigma_en)
	#cov[0:n,0:n] = cov[0:n,0:n] + sigma_delta(xf) 
	
	#new_sigma_delta = tt.set_subtensor(sigma_delta,tt.concatenate([sigma_delta,tt.zeros((n,q),)], axis=1))
	#z_obs = pm.gp.GP('z_obs', cov_func=new_cov, observed={'X':X, 'Y':z})
	mu = np.zeros(n+m)
	z_obs = pm.MvNormal('z_obs', mu=mu, cov=sigma_z, observed=z)

	trace = pm.sample(5000, step = Metropolis(), tune = 5000);
	pm.traceplot(trace);
# l_delta=0.3
# lambda_delta=1.
# sigma_delta = theano.function([], sigma_delta(xf))()



#print(new_cov)
"""

with model:
	trace = pm.sample()


#pm.gelman_rubin(trace)
#pm.traceplot(trace)
#pm.summary(trace)


#plt.show()
x_shared.set_value(xf)
z_shared.set_value(y)    
    
with model:
	post_pred = pm.sample_ppc(trace, model=model, samples=500)

print(post_pred['z_obs'])
x_plot = np.arange(1,n,1)

fig, ax = plt.subplots(figsize=(14,5))
for pred in post_pred:
	ax.plot(x_plot, pred, color=cm(0.3), alpha=0.3)
# overlay the observed data
ax.plot(x_plot,y,'ok',ms=10);
ax.set_xlabel("x");
ax.set_ylabel("f(x)");
ax.set_title("Posterior predictive distribution");

plt.show()


print("END")

"""




