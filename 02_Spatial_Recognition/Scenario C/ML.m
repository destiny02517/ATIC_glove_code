clear all;
close all;

H = [20:1:52]+1;
% 手工剔除某些不需要的值
H(ismember(H, [23, 29, 33, 40])) = [];  
C = length(H);
L = 1251;

Ntest = 60;
Ntrain = 180;
Nval = 60;

for i = 1:1:C
    eval(['X',num2str(i),num2str(i),num2str(i),'=cell2mat(struct2cell(load(''V1_',num2str(H(i)),'.mat'', ''X11'')))'';']);
    eval(['Y',num2str(i),num2str(i),num2str(i),'=cell2mat(struct2cell(load(''V2_',num2str(H(i)),'.mat'', ''X22'')))'';']);
end

% ------------------------------------ run once and save

for i=1:1:C;
    eval(['A',num2str(i),'=randperm(length(X',num2str(i),num2str(i),num2str(i),'(1,:)),Ntest);']);
    eval(['A',num2str(i),'=fliplr(sort(A',num2str(i),'));']);
end

% save('test.mat','A1','A2','A3','A4','A5','A6','A7','A8','A9','A10');



% ------------------------------------ test
% load('test.mat');

for k = 1:1:C
    eval([' x_test',num2str(k),'= X',num2str(k),num2str(k),num2str(k),'(:,A',num2str(k),');']);
    eval([' y_test',num2str(k),'= Y',num2str(k),num2str(k),num2str(k),'(:,A',num2str(k),');']);
end

% ------------------------------------ val
for k = 1:1:C
    for i = 1:1:length(A1)
        eval(['X',num2str(k),num2str(k),num2str(k),'(:,A',num2str(k),'(i))=[];']);
        eval(['Y',num2str(k),num2str(k),num2str(k),'(:,A',num2str(k),'(i))=[];']);
    end
end

for i=1:1:C
    eval(['B',num2str(i),'=randperm(length(X',num2str(i),num2str(i),num2str(i),'(1,:)),Nval);']);
    eval(['B',num2str(i),'=fliplr(sort(B',num2str(i),'));']);
end


for k = 1:1:C
    eval([' x_val',num2str(k),'= X',num2str(k),num2str(k),num2str(k),'(:,B',num2str(k),');']);
    eval([' y_val',num2str(k),'= Y',num2str(k),num2str(k),num2str(k),'(:,B',num2str(k),');']);
end


% ------------------------------------ train
for k = 1:1:C
    for i = 1:1:length(B1)
        eval(['X',num2str(k),num2str(k),num2str(k),'(:,B',num2str(k),'(i))=[];']);
        eval(['Y',num2str(k),num2str(k),num2str(k),'(:,B',num2str(k),'(i))=[];']);
    end
end

for i=1:1:C
    eval(['D',num2str(i),'=randperm(length(X',num2str(i),num2str(i),num2str(i),'(1,:)),Ntrain);']);
    eval(['D',num2str(i),'=fliplr(sort(D',num2str(i),'));']);
end

for k = 1:1:C
    eval([' x_train',num2str(k),'= X',num2str(k),num2str(k),num2str(k),'(:,D',num2str(k),');']);
    eval([' y_train',num2str(k),'= Y',num2str(k),num2str(k),num2str(k),'(:,D',num2str(k),');']);
end

% ------------------------------------- save

for i =  1:1:C
    eval(['x',num2str(i),'=[x_train',num2str(i),',x_val',num2str(i),',x_test',num2str(i),'];']);
    eval(['y',num2str(i),'=[y_train',num2str(i),',y_val',num2str(i),',y_test',num2str(i),'];']);
end

for i =  1:1:C
    w = i;
    % 保存测试数据
    eval(['csvwrite(''E:\桌面\Phd_1\glove\datasheet\object\test\',num2str(i),'.csv'',x',num2str(w),');']);
    eval(['csvwrite(''E:\桌面\Phd_1\glove\datasheet\object\sense\',num2str(i),'.csv'',y',num2str(w),');']);
    
    % 发送保存提示
    disp(['✅ 数据已成功保存: test', num2str(i), '.csv 和 sense', num2str(i), '.csv']);
end