% function EMAlgo(K, epsilon, X)
clear
clc

fig = double(imread('./pic/075.jpg'));%��ȡͼƬ
c1 = fig(:,:,1);%ͨ��1
c2 = fig(:,:,2);%ͨ��2
c3 = fig(:,:,3);%ͨ��3

%��X����ͨ�����ݶ�ʹ��ʱ�������Ƚϲ���һ�¸�ͨ��ͼƬ���Է��֣�
%����ͨ�������ݲ���ر�󣬵���ʹ�ø�������ݣ�������������ν�
%����ͨ�����ݽ��������ֵ�úú��о���

% X =[c1(:),c2(:),c3(:)]; %����X�����ж���
X = [c1(:)]; %X�����У�ÿһ����һ������
K =2; % ������
epsilon = 1e-5;

% ���´������Automatic Corneal Ulcer Segmentation Combining Gaussian Mixture
% Modeling and Otsu Method�е�α�����д

% Initialization
[N, dim] = size(X);% N��dimά�ȵ����أ���ͨ��ͼƬdim=1��rgbͼƬdim=3
t = 0; %��������

% tau, mu, sigmaΪ�����ƵĲ��������������˶�ά����洢�����struct,cellҪ��ʡ��Դ
% tau, mu, sigma��ʼ��ֵ�ǳ���Ҫ
tau = 1/K * ones(1,K);% Mixing coefficient
m = max(X)+1;%ͼƬ����ͨ���������ֵ
mu = (1:K)/(K+1).* ones(dim, K) .* m';% ��ʼ��mu���ȷֲ�
sigma = zeros(dim, dim, K);
for i = 1:K
    sigma(:,:,i) = m .* eye(dim,dim);%��ʼ��sigma������������Ϊ�������ֵ
end

rnk = zeros(N, K); % rnkΪ������ʣ�posterior probability��

% �������˹����,��������һ����������,�ȼ��ڵ�������һ������
% xά��Ϊ N*dim����ÿһ��Ϊһ�����أ�����N��rgb�����أ�xά�ȼ�ΪN*3
% muά��Ϊ 1*dim��sigmaά��Ϊ dim*dim.
% ����ֵά��Ϊ N*1.
GaussianFunc = @(x, mu, sigma)...
            exp(-1/2 * (x-mu).*(inv(sigma)*(x-mu)')')...
            /sqrt((2*pi)^size(x,2)*det(sigma));

        
%��ǰ������Ȼ����ֵ��LogLikelihood�����������ĵ�������
llh = LogLikelihood(X, tau, mu, sigma, GaussianFunc);

while true
    t = t + 1;
    % E step, �ο������еĹ�ʽ��5��
    for class = 1:K
        rnk(:, class) = GaussianFunc(X, mu(:, class)', sigma(:, :, class));
    end
    rnk = (rnk.*tau)./sum(rnk.*tau, 2);
    
    
    % M step���ο������еĹ�ʽ��6������7������8��
    for class = 1:K
        mu(:,class) = (sum(rnk(:, class).*X, 1)./sum(rnk(:, class)))';
        zeroMean = (X-mu(:,class)');
        sigma(:,:,class) = (zeroMean' * (zeroMean .* rnk(:,class)))./sum(rnk(:,class));
    end
    tau = sum(rnk)./N;
    
    % ��ӡ���
    disp(['+++++++++++ iteration ',num2str(t),' ++++++++++++'])
%     disp(['tau: ', num2str(tau)])
%     disp('mu: ')
%     disp(num2str(mu))
%     disp('sigma: ')
%     disp(num2str(sigma))
    
    
    % �˳�ѭ������
    llh_new = LogLikelihood(X, tau, mu, sigma, GaussianFunc);
    if (llh_new - llh)<=epsilon %����Ȼ��������������ʱ���˳�
        break;
    end
    llh = llh_new;
    if(t>100)% ���ߵ�����������������ʱ���˳�
        break
    end
  
end

% ���Էָ�Ч��
% calculate mask
[M,Indx]=max(rnk,[],2);
mask = reshape(Indx,size(c1));
% Convert categories with different colors
coloredLabels = label2rgb (mask, 'hsv', 'k', 'shuffle');
figure,imshow(coloredLabels)


%% ������Ȼ����
function result = LogLikelihood(X, tau, mu, sigma, fcnHandle)
% �ú�����������log-likelihood given in Eq. (4)��
% �����fcnHandle�Ƕ������˹����
    k = length(tau);% �ֶ�����
    [n, ~] = size(X);
    p = zeros(n, k);% ÿһ�����ص�ĸ���
    for j = 1:k
        temp =fcnHandle(X, mu(:,j)', sigma(:,:,j));
        p(:,j) = tau(j) * temp;
    end
    result = sum(log(sum(p, 2)+eps)); %����eps��ȷ������log��ֵ������
end