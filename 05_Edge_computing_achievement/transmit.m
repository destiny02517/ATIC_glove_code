clc;
clear all;
close all; % 关闭之前的图片，防止内存占用过多

% —— 配置区 ——
base_name = 'TPU70';    % 第一级文件夹名
sub_name  = 'sense';  % 第二级文件夹名

% 使用你之前 Python 脚本找出的全局最值
min_val   = 0.658;
max_val   = 0.87139;

% 定义量化的目标范围 (int8)
int8_min  = -128;
int8_max  = 127;

% 获取当前脚本所在的绝对路径
script_dir = fileparts(mfilename('fullpath'));

% 输入路径：...\TPU\locat
in_folder  = fullfile(script_dir, base_name, sub_name);

% —— 修改后的路径逻辑 ——
% 1. 生成名称：TPU locat float to int
out_name   = sprintf('%s %s float to int', base_name, sub_name);

% 2. 核心修改：将输出文件夹放在 base_name (TPU) 文件夹内
% 结果如：...\TPU\TPU locat float to int
out_folder = fullfile(script_dir, base_name, out_name);

% 自动检查并创建输出文件夹
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
    fprintf('✅ 已创建输出文件夹: %s\n', out_folder);
else
    fprintf('ℹ️ 输出文件夹已存在: %s\n', out_folder);
end

% —— 批处理循环 ——
for i = 1:50
    % 构造输入和输出的完整路径
    infile  = fullfile(in_folder, sprintf('%d.csv', i));             
    outfile = fullfile(out_folder, sprintf('%d.csv', i));
    
    % 检查输入文件是否存在
    if ~exist(infile, 'file')
        fprintf('⚠️ 跳过：找不到文件 %s\n', infile);
        continue;
    end
    
    % 1) 读取原始数据
    data = readmatrix(infile);
    
    % 2) 执行线性量化公式
    % 公式：(data - min) / (max - min) * (量化区间长度) + 偏移量
    qdata = round( (data - min_val) / (max_val - min_val) ...
                  * (int8_max - int8_min) + int8_min );
    
    % 3) 边界裁剪（防止 round 之后超出 -128~127）
    qdata = max( min(qdata, int8_max), int8_min );
    qdata = int8(qdata);
    
    % 4) 打印日志
    fprintf('\n--- 处理进度: %d.csv ---\n', i);
    fprintf('   原始数据范围:   [%.5f, %.5f]\n', min(data(:)), max(data(:)));
    fprintf('   量化后数据范围: [%d, %d]\n', min(qdata(:)), max(qdata(:)));
    
    % 5) 绘图验证 (原始 vs 还原)
    % 反量化：看看数据丢了多少精度
    recon = double(qdata - int8_min) / (int8_max - int8_min) ...
          * (max_val - min_val) + min_val;
    
    figure('Name', sprintf('文件 %d 结果对比', i), 'NumberTitle', 'off');
    subplot(2,1,1);
    plot(data(1,:), 'b', 'LineWidth', 1.5); hold on;
    plot(recon(1,:), 'r--', 'LineWidth', 1);
    title(['原始 vs 还原 (第1行) - 文件 ', num2str(i)]);
    legend('Original', 'Reconstructed'); grid on;
    
    subplot(2,1,2);
    plot(qdata(1,:), 'g', 'LineWidth', 1.2);
    title(['量化后的 int8 序列 - 文件 ', num2str(i)]);
    ylabel('Value (-128 to 127)'); grid on;
    
    % 6) 保存量化结果
    writematrix(qdata, outfile);
    fprintf('   💾 结果已保存至: %s\n', outfile);
end

disp('=======================================');
disp('✨ 所有文件量化转换并绘图完成！');