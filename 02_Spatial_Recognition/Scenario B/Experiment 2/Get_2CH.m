% 批量处理 CSV 文件 (1.csv ~ 30.csv)
dt = 0.002;  % 采样间隔 0.002 s

for k = 1:30
    % 输入文件名
    infile = sprintf('%d.csv', k);
    
    % 读取整个 CSV 文件
    data = readmatrix(infile);
    
    % 提取第一列和第六列
    col1 = data(:, 1);
    col6 = data(:, 6);
    
    % 生成时间序列
    t1 = (0:length(col1)-1)' * dt;
    t6 = (0:length(col6)-1)' * dt;
    
    % 合并时间和数据
    out1 = [t1, col1];
    out2 = [t6, col6];
    
    % 保存第一列 -> deal_k_1.csv
    outfile1 = sprintf('deal_%d_1.csv', k);
    writematrix(out1, outfile1);
    
    % 保存第六列 -> deal_k_2.csv
    outfile2 = sprintf('deal_%d_2.csv', k);
    writematrix(out2, outfile2);
    
    fprintf('完成处理: %s -> %s, %s\n', infile, outfile1, outfile2);
end
