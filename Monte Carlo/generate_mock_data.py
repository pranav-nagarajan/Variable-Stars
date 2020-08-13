
np.random.seed(42) # set the random seed so results are reproducible 
sigma_intr = 0.03 # intrinsic scatter term
sigma_noise = 0.02 # observational uncertainty term 

N_gals = 4 # number of galaxies
sig_feh = 0.5 # width of each galaxy's MDF, in dex 

mus = np.random.uniform(18, 25, N_gals) # distance moduli
N_stars = np.random.randint(10, 40, N_gals) # number of stars in each galaxy
fehs_mean = np.random.uniform(-0.7, -2.5, N_gals) # mean of MDF of each galaxy
period_slope, zp, metal_slope = -1.8, -1, 0.25 # "true" parameters of the PWZ

all_fehs, all_P, all_mus, all_galaxy_ids = [], [], [], []
for i in range(N_gals):
    all_fehs.append(fehs_mean[i] + sig_feh*np.random.randn(N_stars[i]))
    all_P.append(10**np.random.uniform(-0.35, -0.55, N_stars[i])) 
    all_mus.append(N_stars[i]*[mus[i]]) 
    all_galaxy_ids.append(N_stars[i]*[i])
all_fehs, all_P, all_mus, all_galaxy_ids = np.concatenate(all_fehs), np.concatenate(all_P), np.concatenate(all_mus), np.concatenate(all_galaxy_ids) 
mags = all_mus + zp + period_slope*np.log10(all_P) + metal_slope*all_fehs + sigma_intr*np.random.randn(len(all_P)) + sigma_noise*np.random.randn(len(all_P))
all_obs_uncertainites = np.ones(len(all_P))*sigma_noise

data =  np.vstack([all_P, mags, all_galaxy_ids, all_fehs, all_obs_uncertainites]).T  
np.savetxt('mock_data_rrl.dat', data, fmt='%.4f', delimiter='   ')

# remember to keep track of what the true distance moduli and mean fehs were. 
# in fitting, you should only use fehs_mean (since in real life you don't know the metallicty of each RRL). But it can be good to keep track of all_fehs so you can test whether you're getting them right. 