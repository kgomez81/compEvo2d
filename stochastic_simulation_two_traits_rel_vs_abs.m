function [v,v1,v2,varx,vary,cov] = stochastic_simulation_two_traits(N,s1,u1,s2,u2,ud1,ud2,steps,...
                                    collect_data,start_time,end_time,outputfile)
% The code below has been modified from the source code made
% availabe by Pearce MT and Fisher DS, obtained from:
% 
% https://datadryad.org/resource/doi:10.5061/dryad.f36v6 
%
% Stochastic simulation for two trait model, with relative vs absolute fitness. 
% Each step is one generation. Rates (such as s,u,r) are per generation. 
% The expected size of each subpop after selection, mutation, and mating is computed.
% If expected size < :cutoff: then the size is drawn from Poisson
% distribution with mean equal to the expected size. Otherwise the
% subpopulation size = expected size. 

% output :v, v1, v2:        total rate of adaptation, and rate of adaptation in traits 1 and 2. 
% output :varx, vary, cov:  time averaged variances and covariance in traits 1 and 2.

% input :N: population size
% input :s1: effect size of beneficial mutation in trait 1 (absolute fitness), reduces the death rate of an individual. 
% input :u1: mutation rate per locus of trait 1 (absolute fitness).
% input :s2: effect size of beneficial mutation in trait 2 (relative fitness), increases the competitive ability of an individual. 
% input :u2: mutation rate per locus of trait 2 (relative fitness).
% input :ud1: deleterious mutation rate per locus of trait 1 (absolute fitness).
% input :ud2: deleterious mutation rate per locus of trait 2 (relative fitness).
% input :steps: number of steps for simulation.
% input :collect_data: true/false - collect detailed data on 2d distr. per generation
% input :start_time: start time for collecting detailed data on 2d distribution
% input :end_time: end time for collecting detailed data on 2d distribution
% input :outputfile: string with filename where detailed data will be stored

digits(16);

% initialize variables
pop=N;                      % abundances of a classes
Na = 0;                     % actual population size
fit=0;                      % total fitness of a class
fitx=0;                     % fitness in trait 1 of a class (absolute fitness trait)
fity=0;                     % fitness in trait 2 of a class (relative fitness trait)
nosefitness = 0;            % total fitness of the front
meanfitness = 0;            % mean absolute fitness of the population
meanfitx = 0;               % mean absolute fitness (may not use)
meanfity = 0;               % mean relative fitness
meanfit_s = 0;              % mean absolute fitness of the population
meanfitx_s = 0;             % mean absolute fitness (may not use)
meanfity_s = 0;             % mean relative fitness 
varx = 0;                   % variance in trait 1
vary = 0;                   % variance in trait 2
cov = 0;                    % covariance between trait 1 and 2
cutoff=10/min(s1,s2);       % population cutoff for stochasticity

if (collect_data)           % store parameters used in simulation
    fileID = fopen([outputfile '-0.txt'],'w');
    fprintf(fileID,'%.16f,%.16f,%.16f,%.16f,%.16f',N, s1, s2, u1, u2);
    fclose(fileID);
    fileID1 = fopen([outputfile '-1.txt'],'w'); %file for all other 2d wave data per generation
    fileID2 = fopen([outputfile '-2.txt'],'w'); %file for data on classes per generation
    fileID3 = fopen([outputfile '-3.txt'],'w'); %file for data on abundances per generation
end

% Main loop for simulation of each generation
for timestep=1:steps   
    
    %%%%%%%%%%%%%%%%%%%
    % Remove columns of zeros of decreasing fitness classes
    while any(pop(:,1))==0
        pop(:,1)=[];
        fit(:,1)=[];
        fity(1)=[];
    end
    
    while any(pop(1,:))==0 
        pop(1,:)=[];
        fit(1,:)=[];
        fitx(1)=[];
    end
    
    % Add columns for padding, i.e. for new class produced by mutations
    dim=size(pop); 
    if any(pop(:,dim(2)))==1    % check for expansion of front in direction of trait 1
        pop(:,dim(2)+1)=zeros(dim(1),1);
        fit(:,dim(2)+1)=fit(:,dim(2))+ones(dim(1),1);
        fity(dim(2)+1)=fity(dim(2))+1;
    end
    
    dim=size(pop); 
    if any(pop(dim(1),:))==1    % check for expansion of front in direction of trait 2
        pop(dim(1)+1,:)=zeros(1,dim(2));
        fit(dim(1)+1,:)=fit(dim(1),:)+ones(1,dim(2));
        fitx(dim(1)+1)=fitx(dim(1))+1;
    end
    
    %%%%%%%%%%%% 
    % Find expected frequencies after selection and mutation
    dim=size(pop);  
    freq=pop/sum(sum(pop));                                 % array with frequencies of the each class
    
    % calculate growth due to selection
    fitx_arry = s1*fitx'*ones(1,dim(2));           % array with # of mutations in trait 1
    fity_arry = s2*ones(dim(1),1)*fity;            % array with # of mutations in trait 2
    fit = fitx_arry + fity_arry;                   % total fitness
    meanfitness = sum(sum(times(freq,fit)));
    
    newfreq=times(exp(fit-meanfitness),freq);  % SELECTION STEP HERE
    newfreq=newfreq/sum(sum(newfreq));          % make sure frequencies still add to one.
    
    % calculate changes in abundances due to mutations
    
    % beneficial mutations
    z1=zeros(1,dim(2));
    mutatex=[z1; newfreq];
    mutatex(dim(1)+1,:)=[];                     % newfreq already has padding, get rid of extra padding from shift
    
    z2=zeros(dim(1),1);
    mutatey=[z2 newfreq];
    mutatey(:,dim(2)+1)=[];                     % newfreq already has padding, get rid of extra padding from shift
    
    % deletirious mutations (note: del mutations in lowest classes ignored)
    z1=zeros(1,dim(2));
    mutatexdel=[newfreq; z1];
    mutatexdel(1,:)=[];                     % newfreq already has padding, get rid of extra padding from shift
    
    z2=zeros(dim(1),1);
    mutateydel=[newfreq z2];
    mutateydel(:,1)=[];                     % newfreq already has padding, get rid of extra padding from shift
    
    nomutate=(1-u1-u2-ud1-ud2)*newfreq;
    postmutate=nomutate+u1*mutatex+u2*mutatey+ud1*mutatexdel+ud2*mutateydel;    
    newfreq=nomutate+postmutate;
    
    % For subpopulations with size less than the stoch_cutoff, draw size
    % from poisson distribution. Otherwise, size = N*(expected frequency).
    
    newpop=N*newfreq;
    stoch=newpop<cutoff;
    stochpop=poissrnd(newpop(stoch));           % sample poisson for abundances below stochasticity cutoff
    newpop(stoch)=stochpop;
    
    newpop=round(newpop);    
    Na = sum(sum(newpop));
    
    meanfitness = sum(sum(times(newpop,fit)))/Na;
    meanfitx = sum(sum(times(newpop,fitx_arry)))/Na;
    meanfity = sum(sum(times(newpop,fity_arry)))/Na;
    
    nosefitness = max(max(times(fit,sign(newpop))));    % calculate most fitness of most fit class
    pop = newpop*(N/Na);
    
    % recompute time-average of variances and covariances
    if timestep > 3000
        varx = (1/timestep)*((timestep-1)*varx + sum(sum(times(newpop,(fitx_arry-meanfitx).^2)))/Na);
        vary = (1/timestep)*((timestep-1)*vary + sum(sum(times(newpop,(fity_arry-meanfity).^2)))/Na);
        cov = (1/timestep)*((timestep-1)*cov + sum(sum(times(newpop,(fitx_arry-meanfitx).*(fity_arry-meanfity))))/Na);
    else
        if timestep==3000
            varx = sum(sum(times(newpop,(fitx_arry-meanfitx).^2)))/Na;
            vary = sum(sum(times(newpop,(fity_arry-meanfity).^2)))/Na;
            cov = sum(sum(times(newpop,(fitx_arry-meanfitx).*(fity_arry-meanfity))))/Na;
            meanfit_s = meanfitness;
            meanfitx_s = meanfitx;
            meanfity_s = meanfity;
        end
    end
    
    if( collect_data && (timestep >= start_time) && (timestep <= end_time) )
        
        % compute the covariance using only classes at the front
        indx_front = (fit==nosefitness);
        meanfitx_front = sum(pop(indx_front).*fitx_arry(indx_front))/sum(pop(indx_front));
        meanfity_front = sum(pop(indx_front).*fity_arry(indx_front))/sum(pop(indx_front));
        front_cov = sum(pop(indx_front).*(fitx_arry(indx_front)-meanfitx_front).*(fity_arry(indx_front)-meanfity_front))/sum(pop(indx_front));
        
        % compute variances, covarainces and population load
        sigmax2 = sum(sum(times(newpop,(fitx_arry-meanfitx).^2)))/Na;
        sigmay2 = sum(sum(times(newpop,(fity_arry-meanfity).^2)))/Na;
        sigmaxy = sum(sum(times(newpop,(fitx_arry-meanfitx).*(fity_arry-meanfity))))/Na;
        pop_load = nosefitness - meanfitness;
        
        % compute v1, v2, sigma12, G eigenvalues and orientation 
        [D,L] = eig([sigmax2 sigmaxy; sigmaxy sigmay2]);
        evec1 = [cosd(45) sind(45); -sind(45) cosd(45)]*(sign(D(1,1))*D(:,1)); 
        Gang = atan2d(evec1(2),evec1(1)); 
        
        % print data to output files, need: times,mean_fit,fit_var,fit_cov,pop_load,dcov_dt,vU_thry,v2U_thry
        fprintf(fileID1,'%i,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f\n',timestep,sigmax2,sigmay2,sigmaxy,front_cov,pop_load,L(2,2),L(1,1),Gang,meanfitness,meanfitx,meanfity);
        
        for i=1:size(pop,1)
            for j=1:size(pop,2)
                if(pop(i,j)>0)
                    fprintf(fileID2,'[%i,%i],',fitx(i),fity(j));
                    fprintf(fileID3,'%f,',pop(i,j));
                end
            end
        end
        
        fprintf(fileID2,'\n');
        fprintf(fileID3,'\n');
    end
    
end

v = (meanfitness-meanfit_s)/(steps-3000);
v1 = (meanfitx-meanfitx_s)/(steps-3000);
v2 = (meanfity-meanfity_s)/(steps-3000);

% close output files
if(collect_data)
    fclose(fileID1);
    fclose(fileID2);
    fclose(fileID3);
end

end

function new_expect_freq = Abs_Rel_Fitness_function(fitness,freq,popsize,birth_rate)
    % Abs_Rel_Fitness calcualtes the expected frequencies due to selection
    % in absolute and relative fitness. Selection occurs according to the
    % formulation given in Bertram and Masel's (2019), varying density
    % lottery model
    
    % parameters and variables involved in selection
    
    delta
    
    new_expect_freq = popsize;
    
end