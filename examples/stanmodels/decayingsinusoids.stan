//from https://discourse.mc-stan.org/t/custom-log-likelihood-function-speed-up/19757
functions {
  real freq_dist_log(vector y,vector time, vector f,vector alpha, int k,int N, 
  real delta, real noise_prior_alpha, real noise_prior_beta){

// set up local values  
  matrix[N,2*k] D_damp; // D-matrix with dampning
  matrix[2*k,2*k] D_damp_ch; // cholesky D-matrix with dampning
  matrix[2*k,N] Sigma; // Sigma cov-matrix
  matrix[N,N] I_mat; // identity matrix
  matrix[N,N] mat1; // matrix made for easier code reading  
  matrix[N,N] L;
  vector[N] A;
  real prob; // log-liklihood 
  
  for (j in 1:k){
    for (i in 1:N){
       D[i,j*2-1]=cos(f[j]*time[i])*exp(-alpha[j]*time[i]);
       D[i,j*2]=sin(f[j]*time[i])*exp(-alpha[j]*time[i]);
    }
  } 
  
// compute Sigma from cholesky decompostion of D
   D_ch=cholesky_decompose(crossprod(D));
   Sigma=delta*D_ch\(D');
  
// set up likelihood function from Sigma 
   
   I_mat=diag_matrix(rep_vector(1,N)); // identity matrix
   mat1=(I_mat+crossprod(Sigma)); // corresponds to (I_N+D*Sigma*D^T)
   L=cholesky_decompose(mat1); // Cholesky decomp of mat1
   A=((L))\y; // solve for A
   
   
  // log-likelihood 
   prob=-0.5*(2*noise_prior_alpha+N)*log(2*noise_prior_beta+A'*A); // return the loglik
 
  return prob;
   
  }
}

data {
  int<lower=0> N; // number of points
  vector[N] y; // observations
  vector[N] time; // time observations
  int <lower=0> k; // no of components
  // priors 
  real<lower=0> noise_prior_alpha; // 
  real<lower=0> noise_prior_beta; // 
 vector[k] alpha_prior; 
  real<lower=0> delta;
}

parameters {
    positive_ordered [k] f  ;
    vector <lower=0> [k] alpha;
}

model {
  // priors
  f~uniform(0,pi());
 
  y~freq_dist(time,f,alpha,k,N,delta,noise_prior_alpha,noise_prior_beta);
}
