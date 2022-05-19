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

clear 
close all
clc

% mex cec14_func.cpp -DWINDOWS
% 17-22 hybrid, not plotted and 29-30...
% 1-3 unimodal func.
% 4-16 multimodal func.
% 23-28 composition func.
func_num=23; 
runs=1; 
D=2; % number of dimension
Xmin=-100;
Xmax=100;

pop_size=10*D;
iter_max=1000;
fhd=str2func('cec14_func');
empty_solution.cost=[];
empty_solution.position=[];
empty_solution.t=[];
solution=repmat(empty_solution,func_num,runs);
lb=Xmin;
ub=Xmax;
X_suru=lb+(ub-lb).*rand(pop_size,D); % initial population

% Optimization with MsRwGWO
[gbest,gbestval,FES,t] = MsRwGWO_func(fhd,D,pop_size,iter_max,Xmin,Xmax,X_suru,func_num);
solution(func_num,1).position = gbest;
solution(func_num,1).cost = abs(gbestval-func_num*100);
solution(func_num,1).t = t;
fprintf('\n---------------------------------------------------------------\n');
fprintf('Optimization with MsRwGWO \n');
fprintf('Func no: %d -> %d. run : best error = %1.2e\n',func_num,runs,solution(func_num,1).cost);

plot_function
drawnow
% Convergence Analysis
FESindex = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]*FES;

% 1. Convergence Analysis
% convergence curve of the selected function
figure (2)
semilogy(FESindex,solution(func_num,1).t,'-db');
xlabel('Function Evaluations');
ylabel('Error Value');
str = sprintf('Convergence Analysis of FN%d',func_num);
title(str);
