%   A Multi-strategy Random weighted Gray Wolf Optimizer            %
%                       (MsRwGWO)                                   %
%																                                   	%
%       A Multi-strategy Random weighted Gray Wolf Optimizer        %
%           for short-term wind speed forecasting                   %
%          Tufan Inac, Emrah Dokur & Ugur Yuzgec                    %
%                Cite this article as follow                        %
% İnaç, T., Dokur, E. & Yüzgeç, U. A multi-strategy random weighted %
% gray wolf optimizer-based multi-layer perceptron model for short- %
% term wind speed forecasting. Neural Comput & Applic (2022).       %
%            https://doi.org/10.1007/s00521-022-07303-4             %

function [gbest,gbestval,fitcount,t]= MsRwGWO_func(fhd,dim,SearchAgents_no,Max_iter,VRmin,VRmax,X_suru,varargin)
rand('seed',sum(100*clock));
gbest_position=zeros(dim*10000/SearchAgents_no,dim);
lu = [VRmin .* ones(1, dim); VRmax .* ones(1, dim)];

% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

if length(VRmin)==1
    VRmin=repmat(VRmin,1,dim);
    VRmax=repmat(VRmax,1,dim);
end

VRmin=repmat(VRmin,SearchAgents_no,1);
VRmax=repmat(VRmax,SearchAgents_no,1);
	
Positions=VRmin+(VRmax-VRmin).*rand(SearchAgents_no,dim);
if dim==2
    Positions=X_suru;
end
fitness=feval(fhd,Positions',varargin{:}); % calculate fitness values...

fitcount=SearchAgents_no;
for i=1:SearchAgents_no  
               
        % Update Alpha, Beta, and Delta
        if fitness(i)<Alpha_score 
            Alpha_score=fitness(i); % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness(i)>Alpha_score && fitness(i)<Beta_score 
            Beta_score=fitness(i); % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness(i)>Alpha_score && fitness(i)>Beta_score && fitness(i)<Delta_score 
            Delta_score=fitness(i); % Update delta
            Delta_pos=Positions(i,:);
        end
end
gbest = Alpha_pos;
gbestval = Alpha_score;
gbest_position(1,:) = gbest;

Distance = abs(Positions(1,1)-Positions(:,1)); 
DistanceSum(1) = sum(Distance)/(SearchAgents_no-1);
% 2. Search History Analysis
if dim==2 % 2d dimension
	refresh = 100;
	figure(21)

% initialize
        x1 = linspace(VRmin(1), VRmax(1), 101);
        x2 = linspace(VRmin(2), VRmax(2), 101);
        x3 = zeros(length(x1), length(x2));
	for i = 1:length(x1)
		for j = 1:length(x2)
			x3(i, j) = feval(fhd,[x1(i);x2(j)],varargin{:});
		end
    end
    
% tiles, labels, legend
	str = sprintf('Search History of FN%d',varargin{:});
	xlabel('x_1'); ylabel('x_2');  title(str);
	contour(x1', x2', x3'); hold on;
	% plot(Positions(:,1),Positions(:,2),'bs','MarkerSize',8);
	plot(Positions(:,1),Positions(:,2),'ko','MarkerSize',6,'MarkerFaceColor','black');
	drawnow
	% plot(gbest(1),gbest(2),'k*','MarkerSize',8);
end
t=[];k=1;
    while fitcount<dim*10000
    old_Positions=Positions; 
	old_fitness=fitness;
    % The parameter 'a' is calculated by using modified equation given below: 
    %----------------------------------------------------------------------------
    a=2*sin((1-k/Max_iter)*(pi/2)); % original -> 2*sin((1-(1:1000)./Max_iter).*(pi/2))+0.5
    %----------------------------------------------------------------------------    
    % Update the Position of search agents including omega wolves...
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; 
            C1=2*r2; 
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j));
            X1=Alpha_pos(j)-A1*D_alpha;
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; 
            C2=2*r2; 
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); 
            X2=Beta_pos(j)-A2*D_beta;      
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a;
            C3=2*r2; 
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); 
            X3=Delta_pos(j)-A3*D_delta;             
           % NEW weighted updating mechanism -> Eqs. (11)-(13)
           %------------------------------------------------------------------------            
            Sum_score=(1/Alpha_score)+(1/Beta_score)+(1/Delta_score);
            w(1)=(1/Alpha_score)/Sum_score;
            w(2)=(1/Beta_score)/Sum_score;
            w(3)=(1/Delta_score)/Sum_score;
            Positions(i,j) = w(1)*X1+w(2)*X2+w(3)*X3;
        end
		%------------------------------------------------------------------------
		%% mutation operator...
		%------------------------------------------------------------------------
		if rand()<0.005
			nmu=ceil(0.1*dim);
			rn=randsample(dim,nmu);
			sigma=0.1*(VRmax(1)-VRmin(1));
			Positions(i,rn)=Positions(i,rn)+sigma.*randn(size(rn))';
		end    
		% NEW boundary checking mechanism...
		%------------------------------------------------------------------------
		Positions(i,:) = boundConstraint(Positions(i,:), old_Positions(i,:),lu); 
		%------------------------------------------------------------------------
    end   
			
	for i=1:size(Positions,1)  
        
        % Calculate objective function for each search agent
		fitness(i)=feval(fhd,Positions(i,:)',varargin{:});
                fitcount=fitcount+1;
	% NEW Greedy Selection mechanism...
		if 	old_fitness(i) < fitness(i)
			Positions(i,:) = old_Positions(i,:); 
			fitness(i) = old_fitness(i);
		end
        % NEW Update Alpha, Beta, and Delta Wolves...
		if fitness(i)<Alpha_score 
			Delta_score=Beta_score; % Update delta
			Delta_pos=Beta_pos;
			Beta_score=Alpha_score; % Update beta
			Beta_pos=Alpha_pos;
			Alpha_score=fitness(i); % Update alpha
			Alpha_pos=Positions(i,:);
		elseif fitness(i)>Alpha_score && fitness(i)<Beta_score 
			Delta_score=Beta_score; % Update delta
			Delta_pos=Beta_pos;
			Beta_score=fitness(i); % Update beta
			Beta_pos=Positions(i,:);
		elseif fitness(i)>Alpha_score && fitness(i)>Beta_score && fitness(i)<Delta_score 
			Delta_score=fitness(i); % Update delta
			Delta_pos=Positions(i,:);
        end  		
    end	

	gbest = Alpha_pos;
	gbestval = Alpha_score;
				
	if fitcount==100*dim||fitcount==200*dim||fitcount==300*dim||fitcount==500*dim...
			||fitcount==1000*dim||fitcount==2000*dim||fitcount==3000*dim||fitcount==4000*dim||fitcount==5000*dim...
			||fitcount==6000*dim||fitcount==7000*dim||fitcount==8000*dim||fitcount==9000*dim||fitcount==10000*dim
		t=[t;abs(gbestval-varargin{:}*100)];              
	end
	k=k+1;
        
	gbest_position(k,:)=gbest;
	Distance = abs(Positions(1,1)-Positions(:,1));
	DistanceSum(k) = sum(Distance)/(SearchAgents_no-1);
        % 2. Search History Analysis
		if dim==2 && (rem(fitcount/SearchAgents_no,refresh) == 0)
			figure(21)
			plot(Positions(:,1),Positions(:,2),'ko','MarkerSize',6,'MarkerFaceColor','black');
			xlabel('x_1'); ylabel('x_2'); title(str);
			drawnow
		end
	end
	
		if dim==2
			% 3. Trajectory Analysis
			figure(12); 
			subplot(2,2,[1 3]);
			contour(x1', x2', x3'); hold on; 
			plot(gbest_position(:,1), gbest_position(:,2), 'ks', 'MarkerSize', 7);
            		hold on 
			plot(gbest_position(end,1), gbest_position(end,2), 'rs', 'MarkerSize', 7);
			str = sprintf('Trajectory of Elite FN%d',varargin{:});
			xlabel('x_1'); ylabel('x_2');  title(str);
			subplot(2,2,2); 
			hold on			
			plot(SearchAgents_no:SearchAgents_no:fitcount,gbest_position(:,1),'r');
			xlabel('Function Evaluations'); ylabel('x_1 position');
            		% legend('GWO','MsRwGWO')
			subplot(2,2,4);
			hold on			
			plot(SearchAgents_no:SearchAgents_no:fitcount,gbest_position(:,2),'r');
			xlabel('Function Evaluations'); ylabel('x_2 position');
            		% legend('GWO','MsRwGWO')
			drawnow
			% 4. Average Distance Analysis
			figure(13);
			hold on			
			plot(SearchAgents_no:SearchAgents_no:fitcount,DistanceSum,'r-');
			xlabel('Function Evaluations');
			ylabel('Average distance');
			str = sprintf('Average Distance of FN%d',varargin{:});
            		title(str);
			% legend('GWO','MsRwGWO')
            		drawnow
		end
end
