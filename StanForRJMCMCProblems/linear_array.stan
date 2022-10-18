data {
  int<lower=0> M;            // number of signals
  int<lower=1> L;            // number of sensors
  int<lower=1> T;            // number of samples for each sensor
  real<lower=0> fs;          // sampling frequency
  real<lower=0> wave_speed;  // wave speed
  vector[T] d[L];            // samples from the sensors 
  real<lower=0> sep;         // separation between sensors
}

transformed data{
  int N = T*L;
  int f_n = T/2;                  // Nyquist frequency
  real N_2 = N/2.0;
  real dTd;
  array[f_n, L, 2] real GHd_factor;
  real sinfunc;
  array[f_n] real sumsinsq;
  array[f_n] real sumcossq;
  array[f_n] real sumsincos;
  
  dTd = 0.0;
  for (l in 1:L){
      dTd = dTd + sum(d[l].*d[l]);
  }
  for (l in 1:L){
      GHd_factor[:, l, 1] = rep_array(0.0, f_n);
      GHd_factor[:, l, 2] = rep_array(0.0, f_n);
  }
  for (omega in 1:f_n){
      // calculate Fourier transform
      for (t in 0:(T-1)){
          for (l in 1:L){
              GHd_factor[omega, l, 1] = GHd_factor[omega, l, 1] + sin(omega*t/fs)*d[l,t+1];
              GHd_factor[omega, l, 2] = GHd_factor[omega, l, 2] + cos(omega*t/fs)*d[l,t+1];
          }
      }
      // calculate other dot products
      sinfunc = sin(4*pi()*omega/fs)/(8*pi()*omega/(T*fs));
      sumsinsq[omega] = T/2 - sinfunc;
      sumcossq[omega] = T/2 + sinfunc;
      sumsincos[omega] = (1-cos(4*pi()*omega/fs))/(8*pi()*omega/(T*fs));
  }
}

parameters {
  real<lower=0, upper=2*pi()> theta[M];  // directions of arrival for each signal 
  real<lower=0> delta2;                  // signal to noise ratio
}

model {
  matrix[M,M] GHG;
  vector[M] GHd;
  real delta2_const;
  real alpha_factor;
  array[M, L] real alpha;
  print("test");
  delta2 ~ inv_gamma(2,0.1);
  delta2_const = 1+1/(N*delta2);
  for (omega in 1:f_n){
      GHd = rep_vector(0.0, M);
      GHG = rep_matrix(rep_row_vector(0.0, M), M);
      for (m in 1:M){
          GHG[m,m] = N;
          alpha_factor = 2*pi()*(omega/fs)*sep*cos(theta[m])/(wave_speed);
          for (l in 1:L){
              alpha[m,l] = (l-1)*alpha_factor;
              GHd[m] = GHd[m] + cos(alpha[m,l])*GHd_factor[omega,l,1] + sin(alpha[m,l])*GHd_factor[omega,l,2];
          }
      }
      for (m in 1:M){
          for (m1 in m+1:M){
              for (l in 1:L){
                  GHG[m,m1] = GHG[m,m1] + sin(alpha[m,l])*sin(alpha[m1,l])*sumcossq[omega] + (cos(alpha[m,1])*sin(alpha[m1,l])+cos(alpha[m1,l])*sin(alpha[m,l]))*sumsincos[omega] + cos(alpha[m,l])*cos(alpha[m1,l])*sumsinsq[omega];
              }
              GHG[m,m1] = delta2_const*GHG[m,m1];
              GHG[m1,m] = GHG[m,m1];
          }
      }
      target += -N_2*log((dTd - (GHd'/GHG)*GHd)/2);
  }
  target += -f_n*log(N*delta2 + delta2_const)/2 +2200000;
}