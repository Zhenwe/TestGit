% function EMAlgo(K, epsilon, X)
clear
clc

fig = double(imread('./pic/075.jpg'));%读取图片
c1 = fig(:,:,1);%通道1
c2 = fig(:,:,2);%通道2
c3 = fig(:,:,3);%通道3

%当X三个通道数据都使用时，结果会比较差，检查一下各通道图片可以发现，
%各个通道的数据差别特别大，导致使用更多的数据，结果反而更差。如何将
%各个通道数据结合起来，值得好好研究。

% X =[c1(:),c2(:),c3(:)]; %变量X可以有多列
X = [c1(:)]; %X变量中，每一行是一个像素
K =2; % 分类数
epsilon = 1e-5;

% 以下代码根据Automatic Corneal Ulcer Segmentation Combining Gaussian Mixture
% Modeling and Otsu Method中的伪代码改写

% Initialization
[N, dim] = size(X);% N个dim维度的像素，单通道图片dim=1，rgb图片dim=3
t = 0; %迭代次数

% tau, mu, sigma为待估计的参数，这里我用了多维矩阵存储，相比struct,cell要节省资源
% tau, mu, sigma初始化值非常重要
tau = 1/K * ones(1,K);% Mixing coefficient
m = max(X)+1;%图片各个通道像素最大值
mu = (1:K)/(K+1).* ones(dim, K) .* m';% 初始化mu均匀分布
sigma = zeros(dim, dim, K);
for i = 1:K
    sigma(:,:,i) = m .* eye(dim,dim);%初始化sigma，将方差设置为最大像素值
end

rnk = zeros(N, K); % rnk为后验概率（posterior probability）

% 多变量高斯函数,这里用了一个内联函数,等价于单独定义一个函数
% x维度为 N*dim，即每一行为一个像素，例如N个rgb的像素，x维度即为N*3
% mu维度为 1*dim，sigma维度为 dim*dim.
% 返回值维度为 N*1.
GaussianFunc = @(x, mu, sigma)...
            exp(-1/2 * (x-mu).*(inv(sigma)*(x-mu)')')...
            /sqrt((2*pi)^size(x,2)*det(sigma));

        
%当前对数似然函数值，LogLikelihood函数定义在文档最下面
llh = LogLikelihood(X, tau, mu, sigma, GaussianFunc);

while true
    t = t + 1;
    % E step, 参考文章中的公式（5）
    for class = 1:K
        rnk(:, class) = GaussianFunc(X, mu(:, class)', sigma(:, :, class));
    end
    rnk = (rnk.*tau)./sum(rnk.*tau, 2);
    
    
    % M step，参考文章中的公式（6）、（7）、（8）
    for class = 1:K
        mu(:,class) = (sum(rnk(:, class).*X, 1)./sum(rnk(:, class)))';
        zeroMean = (X-mu(:,class)');
        sigma(:,:,class) = (zeroMean' * (zeroMean .* rnk(:,class)))./sum(rnk(:,class));
    end
    tau = sum(rnk)./N;
    
    % 打印结果
    disp(['+++++++++++ iteration ',num2str(t),' ++++++++++++'])
%     disp(['tau: ', num2str(tau)])
%     disp('mu: ')
%     disp(num2str(mu))
%     disp('sigma: ')
%     disp(num2str(sigma))
    
    
    % 退出循环条件
    llh_new = LogLikelihood(X, tau, mu, sigma, GaussianFunc);
    if (llh_new - llh)<=epsilon %当似然函数不再有提升时，退出
        break;
    end
    llh = llh_new;
    if(t>100)% 或者当迭代次数超过上限时，退出
        break
    end
  
end

% 测试分割效果
% calculate mask
[M,Indx]=max(rnk,[],2);
mask = reshape(Indx,size(c1));
% Convert categories with different colors
coloredLabels = label2rgb (mask, 'hsv', 'k', 'shuffle');
figure,imshow(coloredLabels)


%% 对数似然函数
function result = LogLikelihood(X, tau, mu, sigma, fcnHandle)
% 该函数用来计算log-likelihood given in Eq. (4)。
% 传入的fcnHandle是多变量高斯函数
    k = length(tau);% 分多少类
    [n, ~] = size(X);
    p = zeros(n, k);% 每一个像素点的概率
    for j = 1:k
        temp =fcnHandle(X, mu(:,j)', sigma(:,:,j));
        p(:,j) = tau(j) * temp;
    end
    result = sum(log(sum(p, 2)+eps)); %加上eps，确保传给log的值大于零
end