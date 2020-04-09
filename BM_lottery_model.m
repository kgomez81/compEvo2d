function new_pop = BM_lottery_model(pop,fit_a,fit_r,L,b,d,sd, sc)
% BM_lottery_model takes in the set of population abundances for each class
% and the fitness array to calculate the the new expected abundances. The
% model is specified in Bertram and Masel 2019 - "Density-dependent 
% selection and the limits of relative fitness" published in TPB
%
% inputs:
% pop = array with current set of abundances 
% fit_a = number of mutations in absolute fitness trait (1st index in pop)
% fit_r = number of mutations in relative fitness trait (2nd index in pop)
% %
% T = total available territories
% b = base birth rate 
% d = base death rate 
% sd = selection coefficient for beneficial mutation descreasing death rate
% sc = selection coefficient for beneficial mutation increasing competitive ability 
% 
% 
% outputs:
% new_pop = expected abundances due to selection
%

% expected change \Delta_+ n_i
Na = sum(sum(pop));     % current population size
U = T - Na;             % unoccupied territories 

m = pop*(U/T);          % array of avg propagules dispersed per class
l = m./U;
L = sum(sum(L));

c_array = ones(size(fit_a,1))*fit_r;
c_bar = m.*c_array./(sum(sum(m));

R = 
A = 

delta_ni = (exp(-L))
end