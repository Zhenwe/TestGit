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

function [Theta,Q,T] = Expectation_Maximization_GMM(K,epsilon,ThetaInit,X)
% K - total number of Gaussian components, reference, K = 2
% epsilon - Termination condition parameter, reference, epsilon = 1e-5
% Theta_0 - include the initial value of Tao, Mu, Sigma; e.g. Theta.Tao
% X - the observed data, each line correspondes a data sample.
%% Initialization
%------Attention���˶δ�����Բ�����ֱ���Ķ�whileѭ�����롣
%------����whileѭ������������£��˶δ�����Ҫ����Ӧ���¡�
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
% Q(1)�����ǳ�ʼֵ������Q(2)������ѭ��������е����������������ѭ��������
% Q(2)��֪����ζ��Theta��2��,T(t=2)�϶���֪

% ��ʱ��t = 1
% calculate Theta(1)
% Theta(1) is known, as the input variable.
% calculate T(t=1)
for i = 1:N
       % calculate the denominator of the T(i,j,t)
       T_den = 0;
       for j = 1:K
           sum_item(j) = Theta(t).Tao(j) * ...
               exp(-0.5*log(det(cell2mat(Theta(t).Sigma(j))))...
               -0.5*d*log(2*pi)...
               -0.5*(X(i,:)' - cell2mat(Theta(t).Mu(j)'))'*...
               cell2mat(Theta(t).Sigma(j))^{-1}*(X(i,:)' ...
               - cell2mat(Theta(t).Mu(j)')));
           T_den = T_den + sum_item(j);
       end
       for j = 1:K
           T_num = Theta(t).Tao(j) * sum_item(j);
           T(i,j,t) = T_num / T_den;
       end
end
% calculate Q(1)
Q(t) = 0;
   for i = 1:N
       for j = 1:K
          Q(t) = Q(t) + T(i,j,t) * (log(Theta(t).Tao(j)) - ...
              0.5*log(det(cell2mat(Theta(t).Sigma(j)))) - ...
              0.5*( (X(i,:)' - cell2mat(Theta(t).Mu(j)')')  ...
              * cell2mat(Theta(t).Sigma(j))^{-1} * (X(i,:)' - ...
              cell2mat(Theta(t).Mu(j)'))  ) - d/2*log(2*pi));
       end
   end
t = t + 1;
% ��ʱ�� t = 2
% calculate Theta(2)
for j = 1:K      
       % Tao - column vector
       Theta(t).Tao(j) = sum(T(:,j,t-1)) / N;
       
       % Mu - column vector
       Temp.Mu = zeros(d,1);
       for i = 1:N
          Temp.Mu = Temp.Mu + T(i,j,t-1) * X(i,:)';  
       end
       Theta(t).Mu(j) = {Temp.Mu / sum(T(:,j,t-1))};
       
       % Sigma - covariance matrix
       Temp.Sigma = zeros(d,d);
       for i = 1:N
          Temp.Sigma = Temp.Sigma + T(i,j,t-1) * (X(i,:)' - ...
              cell2mat(Theta(t).Mu(j))) * (X(i,:)' - ...
              cell2mat(Theta(t).Mu(j)))';  
       end
       Theta(t).Sigma(j) = {Temp.Sigma(j)};
end
% calculate T(t=2)
for i = 1:N
       % calculate the denominator of the T(i,j,t)
       T_den = 0;
       for j = 1:K
           sum_item(j) = Theta(t).Tao(j) * ...
               exp(-0.5*log(det(cell2mat(Theta(t).Sigma(j))))...
               -0.5*d*log(2*pi)...
               -0.5*(X(i,:)' - cell2mat(Theta(t).Mu(j)'))'*...
               cell2mat(Theta(t).Sigma(j))^{-1}*(X(i,:)' ...
               - cell2mat(Theta(t).Mu(j)')));
           T_den = T_den + sum_item(j);
       end
       for j = 1:K
           T_num = Theta(t).Tao(j) * sum_item(j);
           T(i,j,t) = T_num / T_den;
       end
end
% calculate Q(2)
Q(t) = 0;
   for i = 1:N
       for j = 1:K
          Q(t) = Q(t) + T(i,j,t) * (log(Theta(t).Tao(j)) - ...
              0.5*log(det(cell2mat(Theta(t).Sigma(j)))) - ...
              0.5*( (X(i,:)' - cell2mat(Theta(t).Mu(j)')')  ...
              * cell2mat(Theta(t).Sigma(j))^{-1} * (X(i,:)' - ...
              cell2mat(Theta(t).Mu(j)'))  ) - d/2*log(2*pi));
       end
   end
% ���ˣ�Q(1),Q(2),T(t=1),T(t=2),Theta(1),Theta(2)׼����ϣ�����ѭ��
% Attention��������debug�Ķ�����ѭ���д��룬������������Ҫ������



%% Iteration
while norm(Q(t) - Q(t-1)) > epsilon
% Iterations begin from t = 3.
% �Ȳ��ż�����t = t + 1
t = t + 1; % ����ѭ���Ժ�t = 3���ȼ���Theta(3),�����T(t=3)��Q(3) 
 
   % ## M step: maximize Q-function ,update Theta(t)
   % �����ϣ�E step��ǰ��M step�ں�д����ʱӦ��M step��ǰ��E step�ں�
   for j = 1:K      
       % Tao - column vector
       Theta(t).Tao(j) = sum(T(:,j,t-1)) / N;
       
       % Mu - column vector
       Temp.Mu = zeros(d,1);
       for i = 1:N
          Temp.Mu = Temp.Mu + T(i,j,t-1) * X(i,:)';  
       end
       Theta(t).Mu(j) = {Temp.Mu / sum(T(:,j,t-1))};
       
       % Sigma - covariance matrix
       Temp.Sigma = zeros(d,d);
       for i = 1:N
          Temp.Sigma = Temp.Sigma + T(i,j,t-1) * (X(i,:)' - ...
              cell2mat(Theta(t).Mu(j))) * (X(i,:)' - ...
              cell2mat(Theta(t).Mu(j)))';  
       end
       Theta(t).Sigma(j) = {Temp.Sigma(j)};
   end
   
   % ## E step: 
   % calculate T, only need Theta(t)
   % T - T(i,j,t) represents the "mebemership probabilities" 
   for i = 1:N
       % calculate the denominator of the T(i,j,t)
       T_den = 0;
       for j = 1:K
           sum_item(j) = Theta(t).Tao(j) * ...
               exp(-0.5*log(det(cell2mat(Theta(t).Sigma(j))))...
               -0.5*d*log(2*pi)...
               -0.5*(X(i,:)' - cell2mat(Theta(t).Mu(j)'))'*...
               cell2mat(Theta(t).Sigma(j))^{-1}*(X(i,:)' ...
               - cell2mat(Theta(t).Mu(j)')));
           T_den = T_den + sum_item(j);
       end
       for j = 1:K
           T_num = Theta(t).Tao(j) * sum_item(j);
           T(i,j,t) = T_num / T_den;
       end
   end
   % After get the information of Theta(t), calculate the value of Q(t)
   Q(t) = 0;
   for i = 1:N
       for j = 1:K
           % �������ʽ��ȷʵ�Ǽ���Q(t),Ȼ��Theta(t)����֪��
          Q(t) = Q(t) + T(i,j,t) * (log(Theta(t).Tao(j)) - ...
              0.5*log(det(cell2mat(Theta(t).Sigma(j)))) - ...
              0.5*( (X(i,:)' - cell2mat(Theta(t).Mu(j)')')  ...
              * cell2mat(Theta(t).Sigma(j))^{-1} * (X(i,:)' - ...
              cell2mat(Theta(t).Mu(j)'))  ) - d/2*log(2*pi));
       end
   end
      
end
    
    
end





















