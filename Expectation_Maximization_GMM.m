%**************************************************************************
%  Expectation_Maximization_GMM.m
%  Modern Signal Processing (2019 Fall)
%  Project: Eye image Segmentation (color image)
%  Director: Prof. Xiaoying Tang
%  Date: 2019/12/11
%  Author: Team 1
%  Github: https://github.com/zjumhy97/MSP_Fa19_Proj_Team_1
%--------------------------------------------------------------------------
%  Note: this function is according to the wikipeia.
%  https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm
%**************************************************************************

function [Theta] = Expectation_Maximization_GMM(K,epsilon,ThetaInit,d,X)
% K - total number of Gaussian components, reference, K = 2
% epsilon - Termination condition parameter, reference, epsilon = 1e-5
% Theta_0 - include the initial value of Tao, Mu, Sigma; e.g. Theta.Tao
% d - the dimension of x(i)
% X - the observed data, each line correspondes a data sample.
%% Initialization
% t: time index, add 1 in each iteration until termination
t = 1;
% d: the dimension of x(t)
d = length(X(1,:));
% Theta records the information of unknown parameters of GMM during
% iterations, includeing Tao(the weight vector of each Gaussian Model), Mu
% (the mean of each Gaussian Model,vector of d dimension), Sigma (the 
% covariance maatrix of each Gaussian Model, matrix of d*d dimension)
Theta(1) = ThetaInit;
% Q-function: expectation of the log-likelihood function under the
% condition that data X and Theta(t) are known.
Q(1) = ; 

%% Iteration
t = t + 1; % t=2,��������ҪQ(2)��Q(1)����Ϣ���������������ѭ����ʼ����

while norm(Q(t) - Q(t-1)) > epsilon
% Iterations begin from t = 3.
   t = t + 1; % �����Ժ�t = 3�� ��Ҫ����Q(3)����Q(t),�����Q�ļ��㹫ʽҪ��ȥ�������Q(1)��Q(2)
   % E step: calculate Q(t)
   Q(t) = 0;
   
   % Q(t)��Ĳ�̫�ԣ��������ѭ��Ҫ�ٿ�
   for i = 1:N 
      for  j = 1:K % total number of Gaussian components
          sum_item = T_ijt * (log() - 0.5*log() - 0.5*log() - d/2*log(2*pi));
          Q(t) = Q(t) + sum_item;
      end
   end
   
   % M step: maximize Q(t),update Theta(t)
   for j = 1:K
       Theta(t).Tao(j) = sum(T(:,j)) / sum(sum(T));
       % Theta(t).Tao(j) = sum(T(:,j)) / N;
       
       
       % T��tʱ�̵�T����Mu��Sigma����t+1ʱ�̵�
       Temp(j).Mu = zeros(d,1);
       Temp(j).Sigma = zeros(d,d);
       for i = 1:N
          Temp(j).Mu = Temp(j).Mu + T(i,j) * X(i,:)';
          Temp(j).Sigma = Temp(j).Sigma + T(i,j) * (X(i,:)' - Temp(j).Mu)*(X(i,:)' - Temp(j).Mu)';
       end
       
       Theta(t).Mu(j,:) = Temp(j).Mu'/sum(T(:,j)); % line vector
       
       % ����������
       Theta(t).Sigma(j) = Temp(j).Sigma/sum(T(:,j));
   end
   
end
    
    
end





















