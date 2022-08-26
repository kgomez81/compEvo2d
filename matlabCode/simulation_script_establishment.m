% Absolute fitness evolution (no relative fitness) distribution of extinction times versus Te

% basic parameters
T = 1e9;
%b = 1.2;
d = 100/98;
sa = 0.02;
sr = 0.02;
ua = 0.00; %1e-6; originally
uad = 0.00; %1e-5; originally
ur = 0.00;
urd = 0.00;
b=[1.5 1.5 1.5 1.5 1.5 1.5 1.5 1.5   1.82 1.748 1.676 1.604 1.532 1.46 1.388 1.316];
di=[1.54 1.576 1.612 1.648 1.684 1.72 1.756 1.792   1.684 1.684 1.684 1.684 1.684 1.684 1.684 1.684];

nsample=[100 100 100 200 400 500 1000 1000  100 100 100 100 150 300 500 1000];   

s_tEnv = [1e9]; %[92:2:104]
%tEnv = sort([s_tEnv s_tEnv s_tEnv s_tEnv s_tEnv s_tEnv]); %I believe I can just change this term and be fine
EnvR = 1./s_tEnv; % the rate needs to be exceptionally, expectionally, small
% tExt = zeros(size(tEnv));
cutoff = 10/sr; % Ask Kevin

steps = 2e3;
collect_data = 0;
outputfile = 'compEvo2d_data_ml-003';

% pfix check for relative fitness advantage varying b,d
% initial population and fitness data
init_pop = [1e7 1]; %[1e7; 1e6] originally
init_fit_a = round(log(d./di)./log(1+sr)); %set to 1 difference if testing abs fitness case
init_fit_r = [1 2]; % Unsure about the funtionality of this one


parfor i=1:length(b)
    i
    pfix=0;
    for j = 1:nsample(i)
        [pop, fit_a, fit_r] = stochastic_simulation_two_traits_rel_vs_abs_pop( ...
                                        T,init_pop,init_fit_a(i),init_fit_r,b(i),d,sa,ua,uad,sr,ur,urd,EnvR, ...
                                        steps,collect_data,[outputfile '-' num2str(i)]);
        pfix = pfix + (any(pop(fit_r==init_fit_r(2))>1000))/nsample(i);
    end
    dlmwrite('pfix_estimates.dat',[T b(i) di(i) sr nsample(i) pfix],'delimiter',',','precision',16,'-append');
end

                              
