% 获取当前文件夹下所有csv文件 
files = dir('vofa+8.csv');

% 指定输出目录
outputDir = 'E:\桌面\Phd_1\glove\datasheet\30Class_40CH_ADC\40CH_max_min';

% 确保输出目录存在（如果不存在则创建）
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 处理每个csv文件
for i = 1:length(files)
    % 读取当前文件数据
    data = readtable(files(i).name);
    
    % 创建一个新的表格来存储最大值、最小值和列数
    results = table();
    
    % 遍历第2列到第41列
    for col = 1:40
        % 提取第2502行到2602行的数据
        column_data = data{14500:14670, col};
        
        % 找到最大值和最小值
        max_val = max(column_data)* 100;
        min_val = min(column_data)* 100;
        
        % 添加数据到结果表格
        results = [results; table(max_val, min_val, col, 'VariableNames', {'Max_Value', 'Min_Value', 'Column_Index'})];
    end
    
    % 构建完整的输出路径
    [~, baseName, ~] = fileparts(files(i).name);
    outputPath = fullfile(outputDir, [baseName, '_results.xlsx']);
    
    % 将结果保存到指定路径
    writetable(results, outputPath);
end

disp(['所有文件已处理完毕，结果保存在: ' outputDir]);