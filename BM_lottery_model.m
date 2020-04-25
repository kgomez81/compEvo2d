function [newpop,delta_ni, Wi,W_bar,wi,c_bar] = BM_lottery_model(pop,fit_a,fit_r,T,b,d,sa,sr)
% BM_lottery_model takes in the set of population abundances for each class
% and the fitness array to calculate the the new expected abundances. The
% model is specified in Bertram and Masel 2019 - "Density-dependent 
% selection and the limits of relative fitness" published in TPB. Mutations
% are also incorporated in the function since mutations are determined by
% the number of births.
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
% sr = selection coefficient for beneficial mutation increasing competitive ability 
% 
% 
% outputs:
% new_pop = expected abundances due to selection
% delta_ni = change to class ni
% Wi = set of absolute fitness values
% W_bar = mean absolute fitness
% wi = set of relative fitness values (mean of wi = 1)
%

digits(16);

% ------------------- Selection ------------------------------------------

% basic dimensions
ka = size(fit_a,2);
kr = size(fit_r,2);

% expected change \Delta_+ n_i
Na = sum(sum(pop));     % current population size
U = T - Na;             % unoccupied territories 

mi= pop.*(b.*U./T);     % array of avg propagules dispersed per class
li= pop.*(b./T);

L = sum(sum(li));
ci = ones(ka,1)*(1+sr*fit_r);       % relative fitness is 1 + #mut x sc
c_bar = sum(sum(mi.*ci))./(sum(sum(mi)));

inv_di = (((1+sa.*d.*fit_a)./d)').*ones(1,kr);  % calculate inverse because can't have di-->inf
inv_di(inv_di<0) = 0;                           % inverse of death rate should not be negative, set these values to zero 

% NOTE: with inv_di defined above, 1/d_i-1 -1/di = sa, 0<= 1/di <= 1/d, 1 <= di < inf;
if (length(fit_r)==1)
    Ri = (exp(-li)-exp(-L)).*(1-(1+L).*exp(-L))./(L.*(1-exp(-L)));
    Ai = (1-exp(-li)).*(1-(1+L).*exp(-L))./(L.*(1-exp(-L)));

    births = ( exp(-L) + (Ri + Ai).*ci./c_bar ).*li.*U;   % new individuals
else
    Ri = ( c_bar.*exp(-li).*(1-exp(-(L-li))) )./( ci + (c_bar.*L-ci.*li)./(L-li).*(L-1+exp(-L))./(1-(1+L).*exp(-L)) );
    Ai = ( c_bar.*(1-exp(-li)) )./( ci.*li.*(1-exp(-li))./(1-(1+li).*exp(-li)) + (c_bar.*L-ci.*li)./(L-li).*( L.*(1-exp(-L))./(1-(1+L).*exp(-L)) - li.*(1-exp(-li))./(1-(1+li).*exp(-li)) ) );

    % For small li or li=0, Ai will provides NaN. Use Ai_low approximation for low li
    Ai_low = ( c_bar.*li )./( ci + (c_bar.*L-ci.*li)./(L-li).*(L-1+exp(-L))./(1-(1+L).*exp(-L)) );
    Ai(isnan(Ai)) = Ai_low(isnan(Ai));

    births = ( exp(-L) + (Ri + Ai).*ci./c_bar ).*li.*U;   % new individuals
end

newpop = (inv_di).*pop;
delta_ni = (inv_di).*births;

Wi = (newpop + delta_ni)./pop;
W_bar = sum(sum(newpop + delta_ni))/Na;
wi = Wi./W_bar;

end