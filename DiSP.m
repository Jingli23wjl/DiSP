function [Pi_r, Pi_c]=DiSP(A,K)
% Input: A: Adjacency matrix
%        K: number of clusters
% Output: Pi_r: nodes'row community matrix, Pi_r(i,k) is the probability that node i is in row community k
%         Pi_c: nodes'column community matrix, Pi_c(i,k) is the probability that node i is in  column community k
n_r=size(A,1);n_c=size(A,2);[U,~,V]=svds(A,K);[ pure_r ]=SPA(U,K);B_r=U(pure_r,:);Pi_r=U*B_r'/(B_r*B_r');
for i=1:n_r
    if max(Pi_r(i,:))<=0
        Pi_r(i,:)=-Pi_r(i,:);
    end
end
Pi_r=max(0,Pi_r);
for i=1:n_r
    Pi_r(i,:)=Pi_r(i,:)/sum(Pi_r(i,:));
end
[ pure_c ]=SPA(V,K);B_c=V(pure_c,:);Pi_c=V*B_c'/(B_c*B_c');
for i=1:n_c
    if max(Pi_c(i,:))<=0
        Pi_c(i,:)=-Pi_c(i,:);
    end
end
Pi_c=max(0,Pi_c);
for i=1:n_c
    Pi_c(i,:)=Pi_c(i,:)/sum(Pi_c(i,:));
end
end
function [ pure ]=SPA(X,K)
% An implementation of SPA algorithm [1]
% Input: X: n by K data matrix (eigenvectors in our setting), n is the number of nodes, K is the number of communities
% Outpiut: pure: set of pure nodes indices
% [1] Gillis, Nicolas, and Stephen A. Vavasis. "Fast and robust recursive algorithmsfor separable nonnegative matrix factorization." IEEE transactions on pattern analysis and machine intelligence 36.4 (2013): 698-714.
pure = [];
row_norm = vecnorm(X').^2';
for i = 1:K
    [~,idx_tmp] = max(row_norm);pure = [pure idx_tmp];U(i,:) = X(idx_tmp,:);
    for j = 1 : i-1
        U(i,:) = U(i,:) - U(i,:)*(U(j,:)'*U(j,:));
    end
    U(i,:) = U(i,:)/norm(U(i,:)); row_norm = row_norm - (X*U(i,:)').^2;
end
end