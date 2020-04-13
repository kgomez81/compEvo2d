% Absolute fitness evolution (no relative fitness) distribution of extinction times versus Te

% basic parameters
T = 1e9;
b = 1.2;
d = 100/98;
sa = 0.02;
sr = 0.02;
ua = 1e-6;
uad = 0;
ur = 0;
urd = 0;

s_tEnv = [10:2:92];
tEnv = sort([s_tEnv s_tEnv s_tEnv]);
EnvR = 1./tEnv;
tExt = zeros(size(tEnv));
cutoff = 10/sr;

steps = 100000;
collect_data = 1;
outputfile = 'data/compEvo2d_data_ml-001'; ;

% initial population and fitness data
init_pop = [1e7; 1e6];
init_fit_a = [-15 -14];
init_fit_r = [1];

tic
for i = 1:length(tEnv)
    tExt(i) = stochastic_simulation_two_traits_rel_vs_abs( ...
                                    T,init_pop,init_fit_a,init_fit_r,b,d,sa,ua,uad,sr,ur,urd,EnvR(i), ...
                                    steps,collect_data,[outputfile '-' num2str(i)]);
    [i tEnv(i) tExt(i)]
end
toc
dlmwrite('data/compEvo2d_data_ext_times_ml-001.dat',[tEnv' tExt'],'delimiter',',','precision',16);

% ----------------------------------------------------------------------------
% Main Script to run simulations of 2d abs vs rel simulations

% basic parameters
T = 1e9;
b = 1;
d = 1;
sa = 0.02;
sr = 0.02;
ua = 1e-6;
uad = 1e-5;
ur = 1e-5;
urd = 1e-5;

s_tEnv = [10:2:92];
tEnv = sort([s_tEnv s_tEnv s_tEnv]);
EnvR = 1./tEnv;
tExt = zeros(size(tEnv));
cutoff = 10/sr;

steps = 100000;
collect_data = 1;
outputfile = 'data/compEvo2d_data_wR_ml-002';

% initial population and fitness data
init_pop = [1e7; 1e6];
init_fit_a = [-15 -14];
init_fit_r = [1];

tic
for i = 1:length(tEnv)
    tExt(i) = stochastic_simulation_two_traits_rel_vs_abs( ...
                                    T,init_pop,init_fit_a,init_fit_r,b,d,sa,ua,uad,sr,ur,urd,EnvR(i), ...
                                    steps,collect_data,[outputfile '-' num2str(i)]);
    [i tEnv(i) tExt(i)]
end
toc
dlmwrite('data/compEvo2d_data_ext_times_wR_ml-002.dat',[tEnv' tExt'],'delimiter',',','precision',16);
                                
