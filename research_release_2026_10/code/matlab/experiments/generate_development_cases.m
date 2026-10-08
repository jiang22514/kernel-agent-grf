function generate_development_cases(destination)
% Reconstruct the original four synthetic realizations (MATLAB rng(1)).
if nargin<1
    root=fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    destination=fullfile(root,'recomputed','development_data');
end
if ~isfolder(destination),mkdir(destination);end
rng(1);n=100;s=0.1;
x1=rand(n,2);y1=x1(:,2)+0.1*x1(:,1)+s*randn(n,1);
x2=rand(n,2);y2=x2(:,2).^2+x2(:,1).*x2(:,2)+s*randn(n,1);
x3=rand(n,2);y3=sin(x3(:,2))+x3(:,1)+s*randn(n,1);
x4=rand(n,3);y4=x4(:,1).*x4(:,2)+x4(:,2).^2+x4(:,3).^2+s*randn(n,1);
xx={x1,x2,x3,x4};yy={y1,y2,y3,y4};
for i=1:4
    x_raw=xx{i};y=yy{i};p=fullfile(destination,sprintf('EX%d.mat',i));
    assert(~isfile(p),'Refusing to overwrite a generated data file');
    save(p,'x_raw','y');
end
end
