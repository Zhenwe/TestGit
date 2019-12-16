%**************************************************************************
%  getTheta_kmeans.m
%  Modern Signal Processing (2019 Fall)
%  Project: Eye image Segmentation (color image)
%  Director: Prof. Xiaoying Tang
%  Date: 2019/12/16
%  Author: Team 1
%  Github: https://github.com/zjumhy97/MSP_Fa19_Proj_Team_1
%**************************************************************************

function [Theta,mu] = getTheta_kmeans(K,X)
% ���������ȷ��������ʱ��Ҫ�ã����������ٿ�



row_num = size(X,1);
n_channel = size(X,2);

[labels,mu] = kmeans(X,K);
%��MLE�����ʼ��Ȩ�أ�pai���ͷ���sigma
for k = 1:K
    Theta.Tao(k) = sum(labels == k)/row_num; 
    Theta.Sigma{k} = cov(X(labels == k,:)); 
end


end