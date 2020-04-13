function [newpop,newfit_a,newfit_r] = two_dim_abs_vs_rel_mutations(pop,delta_ni,fit_a,fit_r,ua,uad,ur,urd,cutoff)
% two_dim_abs_vs_rel_mutations takes in the existing set of abundances and
% provides the new set of abundances after mutations. Mutations in absolute
% fitness dimension are bounded above by 0. 
%
% The array for pop must be passed in with a buffer (extra row & column) of
% zeros for mutations below to be calculated. 

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
% ua = beneficial mutation rate in absolute fitness trait
% uad = deleterious mutation rate in absolute fitness trait
% ur = beneficial mutation rate in relative fitness trait
% urd = selection mutation rate in absolute fitness trait
% 
% 
% outputs:
% new_pop = expected abundances due to mutations
% fit_a = new set of absolute fitness mutations
% fit_r = new set of relative fitness mutations

% ------------------- Mutations ------------------------------------------
% -------------------------------------------------------------------------
% This part of the code adds rows and columns to hold the influx of
% beneficial and deleterious mutations ahead and behind the distribution
    
% expansion ahead in relative fitness dimension
dim=size(pop);
if any(pop(:,dim(2)))==1    % check expansion of front 
    pop(:,dim(2)+1)=zeros(dim(1),1);
    delta_ni(:,dim(2)+1)=zeros(dim(1),1);
    fit_r(dim(2)+1)=fit_r(dim(2))+1;
end

% expansion ahead in absolute fitness dimension
dim=size(pop); 
if any(pop(dim(1),:))==1    % check expansion of front
    pop(dim(1)+1,:)=zeros(1,dim(2));
    delta_ni(dim(1)+1,:)=zeros(1,dim(2));
    fit_a(dim(1)+1)=fit_a(dim(1))+1; 
end

% expansion behind in the relative fitness dimension
dim=size(pop);
pop = [zeros(dim(1),1) pop];                % expand behind
delta_ni = [zeros(dim(1),1) delta_ni];
fit_r = [fit_r(1)-1 fit_r];

% expansion behind in the absolute fitness dimension
dim=size(pop);
pop = [zeros(1,dim(2)); pop];
delta_ni = [zeros(1,dim(2)); delta_ni];
fit_a = [fit_a(1)-1 fit_a];

% -------------------------------------------------------------------------

% beneficial and deleterious mutations in RELATIVE fitness dimension
dim = size(delta_ni);

% setup array of mutation fluxes in abs fit. running out of mutations (RM)
UbRM = (-(fit_a-1)')*ones(1,dim(2));
UbRM(UbRM <= 0) = 0.*UbRM(UbRM <= 0);
UdRM = (-fit_a')*ones(1,dim(2));         
UdRM(UdRM > 0) = ones(size(UdRM(UdRM > 0)));
UdRM(UdRM <= 0) = 0.*UdRM(UdRM <= 0);

mutate_ben_r = ur.*[zeros(dim(1),1) delta_ni(:,1:end-1)];
mutate_del_r = urd.*[delta_ni(:,2:end) zeros(dim(1),1)];
mutate_ben_a = (ua.*UbRM).*[zeros(1,dim(2)); delta_ni(1:end-1,:)];
mutate_del_a = (uad.*UdRM).*[delta_ni(2:end,:); zeros(1,dim(2))];

% setup multiples of mutation rates to subtract from delta_ni births
Urb_flux = sign(delta_ni);
Urd_flux = sign(delta_ni);
Uab_flux = ((-fit_a')*ones(1,dim(2)));
Uab_flux(Uab_flux <= 0) = 0.*Uab_flux(Uab_flux <= 0);
Uab_flux = Uab_flux.*sign(delta_ni);
Uad_flux = ((-fit_a')*ones(1,dim(2))).*sign(delta_ni);
Uad_flux(Uad_flux >= 0) = ones(size(Uad_flux(Uad_flux >= 0)));
Uad_flux(Uad_flux < 0) = 0.*Uad_flux(Uad_flux < 0);
Uad_flux = Uad_flux.*sign(delta_ni);

% specify beneficial and deleterious mutation fluxes for abs. and rel.
Urb_flux = ur.*Urb_flux;
Urd_flux = urd.*Urd_flux;
Uab_flux = ua.*Uab_flux;
Uad_flux = uad.*Uad_flux;

nomutate = pop + delta_ni.*( 1 - Urb_flux - Urd_flux - Uab_flux - Uad_flux );
postmutate = mutate_ben_r + mutate_del_r + mutate_ben_a + mutate_del_a;    
pop = nomutate + postmutate;
    
% drift cutoff
stoch = (pop<cutoff);
pop(stoch) = poissrnd(pop(stoch));
pop = round(pop);

% clean up classes ahead and behind of distribution
if ((~isempty(pop)) && (sum(sum(pop))>0))
    while any(pop(end,:))==0 
        pop(end,:)=[];
        fit_a(end)=[];
    end
    
    while any(pop(:,end))==0 
        pop(:,end)=[];
        fit_r(end)=[];
    end
    
    while any(pop(1,:))==0 
        pop(1,:)=[];
        fit_a(1)=[];
    end
    
    while any(pop(:,1))==0 
        pop(:,1)=[];
        fit_r(1)=[];
    end
end

newpop = pop;
newfit_a = fit_a;
newfit_r = fit_r;

end