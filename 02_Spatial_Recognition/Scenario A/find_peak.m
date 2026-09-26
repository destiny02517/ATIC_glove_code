% clear; close all;
% 
% % for H = setdiff(1:21, 12)
% for H = 0:20
%     %% 1. 读取 CSV 数据
%     data1 = readmatrix(sprintf('scope_%d_2.csv', H), 'NumHeaderLines', 2);
%     data2 = readmatrix(sprintf('scope_%d_1.csv', H), 'NumHeaderLines', 2);
%     
%     T1 = data1(:,1);
%     V1 = data1(:,2);
%     V2 = data2(:,2);  % 后续用于提取第二通道信号
%     
%     Tg = mean(diff(T1));  % 采样周期
%     fs = 1/Tg;            % 采样频率
%     
%     %% 2. 预滤波：工频带阻滤波（45–55 Hz）
%     frq = [45, 55] / (fs/2);
%     [b1, a1] = butter(2, frq, 'stop');
%     V1_F = filter(b1, a1, V1);
%     % 若需要，也可对 V2 进行相同滤波：
%     V2_F = filter(b1, a1, V2);
%     
%     %% 3. 初步平滑：5点移动平均
%     V1_smooth = movmean(V1_F, 5);
%     
%     %% 4. 包络提取及二次平滑
%     env = abs(hilbert(V1_smooth));          % 取包络
%     win_length = round(0.02 / Tg);           % 约20ms的滑窗
%     env_smooth = movmean(env, win_length);    % 包络平滑
%     
%     %% 5. 自适应阈值设定
%     meanEnv = mean(env_smooth);
%     stdEnv  = std(env_smooth);
%     threshold = meanEnv + 0.2 * stdEnv;       % 自适应阈值
%     
%     %% 6. 峰值检测
%     f = 6.5;
%     MinPeakDistance = round((1/f)/Tg);        % 根据周期设置最小峰间距
%     MinPeakProminence = 0.2 * stdEnv;           % 设置峰值突出度要求
%     
%     [P, L] = findpeaks(V1_smooth, ...
%         'MinPeakDistance', MinPeakDistance, ...
%         'MinPeakHeight', threshold, ...
%         'MinPeakProminence', MinPeakProminence);
%     
%     %% 7. 后续剔除：去除边界和极值点
%     % 剔除开头和末尾的 10 个峰（防止边界效应）
%     if numel(L) > 20
%         L = L(11:end-10);
%         P = P(11:end-10);
%     elseif numel(L) > 10
%         L = L(11:end);
%         P = P(11:end);
%     end
%     
%     % 当峰数足够时，剔除最高和最低各 10 个峰
%     if numel(P) >= 20
%         top10 = maxk(P, 10);           % 最大的10个峰
%         idx_max = find(ismember(P, top10));
%         bot10 = mink(P, 10);           % 最小的10个峰
%         idx_min = find(ismember(P, bot10));
%         remove_idx = unique([idx_max; idx_min]);
%         L(remove_idx) = [];
%         P(remove_idx) = [];
%     end
%     
%     %% 8. 可视化检查
%     figure;
%     plot(T1, V1_F, 'b'); hold on;
%     plot(T1, V1_smooth, 'g');
%     plot(T1(L), P, '*r');
%     title(sprintf('H = %d: 检测到的峰值', H));
%     legend('滤波信号', '平滑信号', '检测峰值');
%     
%     %% 9. 分段提取信号片段并保存
%     % 设置截取参数
%     d1 = 500;
%     d2 = 500;
%     D  = d1 + d2 + 1;      % 片段总长度 = 501
%     
%     N = numel(L);
%     W = NaN(N, D);
%     B = NaN(N, D);
%     
%     for i = 1:N
%         idx = L(i)-d1 : L(i)+d2;
%         % 检查是否越界
%         if idx(1) > 0 && idx(end) <= numel(V1)
%             W(i,:) = V1(idx);
%             B(i,:) = V2(idx);  % 使用原始 V2 或者 V2_F 均可
%         end
%     end
%     
%     % 保存分段结果（文件名根据需要自行调整）
%     save(sprintf('V1_%d.mat', H+1), 'W');
%     save(sprintf('V2_%d.mat', H+1), 'B');
% end
% 






function waveform_viewer()
    % 波形查看和d值设置工具
    % 用于确定合适的d1和d2值
    
    %% 初始化参数
    H = 13; % 默认查看H=13的数据，你可以修改这个值
    d1 = 300; % 峰值前的点数
    d2 = 400; % 峰值后的点数
    
    %% 读取数据并进行峰值检测
    try
        data1 = readmatrix(sprintf('scope_%d_2.csv', H), 'NumHeaderLines', 2);
        data2 = readmatrix(sprintf('scope_%d_1.csv', H), 'NumHeaderLines', 2);
    catch
        error('无法读取文件 scope_%d_2.csv 或 scope_%d_1.csv，请检查文件是否存在', H, H);
    end
    
    T1 = data1(:,1);
    V1 = data1(:,2);
    V2 = data2(:,2);
    Tg = mean(diff(T1));
    fs = 1/Tg;
    
    %% 信号预处理（使用改进的50Hz滤除方法）
    function filtered_signal = remove_50hz_interference(signal, fs)
        sig = signal;
        
        % 50Hz基频带阻滤波器
        if 50 < fs/2
            Q1 = 35; % 高Q值确保窄带阻
            wo1 = 50/(fs/2);
            bw1 = wo1/Q1;
            [b1, a1] = iirnotch(wo1, bw1);
            sig = filtfilt(b1, a1, sig);
        end
        
        % 100Hz二次谐波带阻
        if 100 < fs/2
            Q2 = 30;
            wo2 = 100/(fs/2);
            bw2 = wo2/Q2;
            [b2, a2] = iirnotch(wo2, bw2);
            sig = filtfilt(b2, a2, sig);
        end
        
        % 低通滤波器去除高频噪声
        fc = 200;
        if fc < fs/2
            [b3, a3] = butter(4, fc/(fs/2), 'low');
            sig = filtfilt(b3, a3, sig);
        end
        
        filtered_signal = sig;
    end
    
    % 应用改进的滤波
    V1_F = remove_50hz_interference(V1, fs);
    
    % 去趋势和平滑
    V1_detrend = detrend(V1_F);
    smooth_window = max(3, round(0.005 / Tg));
    V1_smooth = smoothdata(V1_detrend, 'movmean', smooth_window);
    
    %% 峰值检测
    f = 6.5;
    expected_period = round((1/f)/Tg);
    
    % 自适应阈值
    window_size = round(2 * expected_period);
    local_mean = movmean(abs(V1_smooth), window_size);
    local_std = movstd(V1_smooth, window_size);
    adaptive_threshold = local_mean + 0.5 * local_std;
    
    % 峰值检测
    MinPeakDistance1 = round(0.6 * expected_period);
    MinPeakDistance1 = min(MinPeakDistance1, length(V1_smooth) - 1);
    MinPeakDistance1 = max(1, MinPeakDistance1);
    
    [peaks1, locs1, ~, prominences1] = findpeaks(V1_smooth, ...
        'MinPeakDistance', MinPeakDistance1, ...
        'MinPeakProminence', 0.3 * std(V1_smooth));
    
    % 过滤峰值
    valid_idx = peaks1 > adaptive_threshold(locs1);
    P = peaks1(valid_idx);
    L = locs1(valid_idx);
    
    %% 创建交互式界面
    fig = figure('Position', [100, 100, 1400, 800], 'Name', '波形查看和d值设置工具');
    
    % 创建控制面板
    control_panel = uipanel('Parent', fig, 'Position', [0.02, 0.02, 0.25, 0.96], ...
        'Title', '控制面板', 'FontSize', 12);
    
    % H值选择
    uicontrol('Parent', control_panel, 'Style', 'text', ...
        'Position', [10, 350, 80, 20], 'String', 'H值:', 'FontSize', 10);
    h_edit = uicontrol('Parent', control_panel, 'Style', 'edit', ...
        'Position', [100, 350, 60, 25], 'String', num2str(H), 'FontSize', 10);
    
    % d1值设置
    uicontrol('Parent', control_panel, 'Style', 'text', ...
        'Position', [10, 310, 80, 20], 'String', 'd1 (前):', 'FontSize', 10);
    d1_edit = uicontrol('Parent', control_panel, 'Style', 'edit', ...
        'Position', [100, 310, 60, 25], 'String', num2str(d1), 'FontSize', 10);
    
    % d2值设置
    uicontrol('Parent', control_panel, 'Style', 'text', ...
        'Position', [10, 270, 80, 20], 'String', 'd2 (后):', 'FontSize', 10);
    d2_edit = uicontrol('Parent', control_panel, 'Style', 'edit', ...
        'Position', [100, 270, 60, 25], 'String', num2str(d2), 'FontSize', 10);
    
    % 峰值选择
    uicontrol('Parent', control_panel, 'Style', 'text', ...
        'Position', [10, 230, 80, 20], 'String', '峰值编号:', 'FontSize', 10);
    peak_edit = uicontrol('Parent', control_panel, 'Style', 'edit', ...
        'Position', [100, 230, 60, 25], 'String', '1', 'FontSize', 10);
    
    % 显示总峰值数
    peak_count_text = uicontrol('Parent', control_panel, 'Style', 'text', ...
        'Position', [10, 200, 150, 20], 'String', sprintf('总峰值数: %d', length(L)), 'FontSize', 10);
    
    % 按钮
    update_btn = uicontrol('Parent', control_panel, 'Style', 'pushbutton', ...
        'Position', [20, 160, 120, 30], 'String', '更新显示', 'FontSize', 10);
    
    reload_btn = uicontrol('Parent', control_panel, 'Style', 'pushbutton', ...
        'Position', [20, 120, 120, 30], 'String', '重新加载数据', 'FontSize', 10);
    
    save_params_btn = uicontrol('Parent', control_panel, 'Style', 'pushbutton', ...
        'Position', [20, 80, 120, 30], 'String', '保存参数', 'FontSize', 10);
    
    % 信息显示
    info_text = uicontrol('Parent', control_panel, 'Style', 'text', ...
        'Position', [10, 20, 150, 50], 'String', '', 'FontSize', 9, ...
        'HorizontalAlignment', 'left', 'Max', 3);
    
    % 创建绘图区域
    plot_panel = uipanel('Parent', fig, 'Position', [0.28, 0.02, 0.7, 0.96], ...
        'Title', '波形显示', 'FontSize', 12);
    
    % 子图1: 完整信号和峰值
    ax1 = subplot(3, 1, 1, 'Parent', plot_panel);
    % 子图2: 单个波形段
    ax2 = subplot(3, 1, 2, 'Parent', plot_panel);
    % 子图3: 叠加的多个波形段
    ax3 = subplot(3, 1, 3, 'Parent', plot_panel);
    
    %% 回调函数
    function update_display()
        try
            % 获取当前参数
            current_H = str2double(get(h_edit, 'String'));
            current_d1 = str2double(get(d1_edit, 'String'));
            current_d2 = str2double(get(d2_edit, 'String'));
            current_peak = str2double(get(peak_edit, 'String'));
            
            % 检查参数有效性
            if current_peak < 1 || current_peak > length(L)
                set(info_text, 'String', sprintf('峰值编号超出范围!\n有效范围: 1-%d', length(L)));
                return;
            end
            
            % 绘制完整信号
            cla(ax1);
            plot(ax1, T1, V1_F, 'b', 'LineWidth', 0.8); hold(ax1, 'on');
            plot(ax1, T1, V1_smooth, 'g', 'LineWidth', 1.2);
            plot(ax1, T1(L), P, '*r', 'MarkerSize', 8, 'LineWidth', 2);
            
            % 高亮当前选择的峰值
            plot(ax1, T1(L(current_peak)), P(current_peak), 'or', 'MarkerSize', 12, 'LineWidth', 3);
            
            title(ax1, sprintf('H = %d: 完整信号 (当前峰值: %d/%d)', current_H, current_peak, length(L)));
            legend(ax1, '滤波信号', '平滑信号', '所有峰值', '当前峰值', 'Location', 'best');
            grid(ax1, 'on');
            xlabel(ax1, '时间 (s)');
            ylabel(ax1, '幅值');
            
            % 绘制单个波形段
            peak_idx = L(current_peak);
            start_idx = peak_idx - current_d1;
            end_idx = peak_idx + current_d2;
            
            if start_idx > 0 && end_idx <= length(V1)
                % 提取波形段
                wave_indices = start_idx:end_idx;
                wave_time = T1(wave_indices);
                wave_V1 = V1(wave_indices);
                wave_V2 = V2(wave_indices);
                
                % 相对时间（以峰值为0点）
                relative_time = (wave_indices - peak_idx) * Tg * 1000; % 转换为毫秒
                
                cla(ax2);
                plot(ax2, relative_time, wave_V1, 'b', 'LineWidth', 1.5); hold(ax2, 'on');
                plot(ax2, relative_time, wave_V2, 'r', 'LineWidth', 1.5);
                plot(ax2, 0, V1(peak_idx), 'ok', 'MarkerSize', 10, 'MarkerFaceColor', 'yellow');
                
                title(ax2, sprintf('单个波形段 (峰值 %d, 长度: %d+%d+1=%d点)', ...
                    current_peak, current_d1, current_d2, current_d1+current_d2+1));
                legend(ax2, 'V1信号', 'V2信号', '峰值位置', 'Location', 'best');
                grid(ax2, 'on');
                xlabel(ax2, '相对时间 (ms)');
                ylabel(ax2, '幅值');
                
                % 绘制多个波形段叠加
                cla(ax3);
                colors = lines(min(10, length(L))); % 最多显示10个波形
                display_peaks = min(10, length(L));
                
                for i = 1:display_peaks
                    try
                        temp_peak_idx = L(i);
                        temp_start = temp_peak_idx - current_d1;
                        temp_end = temp_peak_idx + current_d2;
                        
                        if temp_start > 0 && temp_end <= length(V1)
                            temp_indices = temp_start:temp_end;
                            temp_relative_time = (temp_indices - temp_peak_idx) * Tg * 1000;
                            temp_wave = V1(temp_indices);
                            
                            plot(ax3, temp_relative_time, temp_wave, 'Color', colors(i,:), 'LineWidth', 1);
                            hold(ax3, 'on');
                        end
                    catch
                        % 跳过无效的波形段
                        continue;
                    end
                end
                
                % 高亮当前波形
                plot(ax3, relative_time, wave_V1, 'k', 'LineWidth', 3);
                
                title(ax3, sprintf('波形叠加显示 (黑色为当前波形，显示前%d个波形)', display_peaks));
                grid(ax3, 'on');
                xlabel(ax3, '相对时间 (ms)');
                ylabel(ax3, '幅值');
                
                % 更新信息显示
                wave_duration = (current_d1 + current_d2 + 1) * Tg * 1000; % 毫秒
                set(info_text, 'String', sprintf('波形信息:\n长度: %d点\n时长: %.1f ms\n采样率: %.1f Hz', ...
                    current_d1+current_d2+1, wave_duration, fs));
                
            else
                set(info_text, 'String', sprintf('警告:\n当前d值会导致\n波形超出信号范围!\n信号长度: %d点', length(V1)));
                cla(ax2);
                text(ax2, 0.5, 0.5, '波形超出范围!', 'HorizontalAlignment', 'center', 'FontSize', 14);
                cla(ax3);
                text(ax3, 0.5, 0.5, '波形超出范围!', 'HorizontalAlignment', 'center', 'FontSize', 14);
            end
            
        catch ME
            set(info_text, 'String', sprintf('错误:\n%s', ME.message));
        end
    end
    
    function reload_data()
        try
            new_H = str2double(get(h_edit, 'String'));
            
            % 重新读取数据
            data1 = readmatrix(sprintf('scope_%d_2.csv', new_H), 'NumHeaderLines', 2);
            data2 = readmatrix(sprintf('scope_%d_1.csv', new_H), 'NumHeaderLines', 2);
            
            % 更新全局变量
            T1 = data1(:,1);
            V1 = data1(:,2);
            V2 = data2(:,2);
            Tg = mean(diff(T1));
            fs = 1/Tg;
            
            % 重新进行信号处理和峰值检测
            V1_F = remove_50hz_interference(V1, fs);
            
            V1_detrend = detrend(V1_F);
            smooth_window = max(3, round(0.005 / Tg));
            V1_smooth = smoothdata(V1_detrend, 'movmean', smooth_window);
            
            f = 6.5;
            expected_period = round((1/f)/Tg);
            window_size = round(2 * expected_period);
            local_mean = movmean(abs(V1_smooth), window_size);
            local_std = movstd(V1_smooth, window_size);
            adaptive_threshold = local_mean + 0.5 * local_std;
            
            MinPeakDistance1 = round(0.6 * expected_period);
            MinPeakDistance1 = min(MinPeakDistance1, length(V1_smooth) - 1);
            MinPeakDistance1 = max(1, MinPeakDistance1);
            
            [peaks1, locs1, ~, prominences1] = findpeaks(V1_smooth, ...
                'MinPeakDistance', MinPeakDistance1, ...
                'MinPeakProminence', 0.3 * std(V1_smooth));
            
            valid_idx = peaks1 > adaptive_threshold(locs1);
            P = peaks1(valid_idx);
            L = locs1(valid_idx);
            
            % 更新界面
            set(peak_count_text, 'String', sprintf('总峰值数: %d', length(L)));
            set(peak_edit, 'String', '1');
            H = new_H;
            
            update_display();
            
        catch ME
            set(info_text, 'String', sprintf('加载失败:\n%s', ME.message));
        end
    end
    
    function save_parameters()
        current_d1 = str2double(get(d1_edit, 'String'));
        current_d2 = str2double(get(d2_edit, 'String'));
        current_H = str2double(get(h_edit, 'String'));
        
        % 保存到文件
        params.d1 = current_d1;
        params.d2 = current_d2;
        params.H = current_H;
        params.total_length = current_d1 + current_d2 + 1;
        params.save_time = datestr(now);
        
        save('waveform_parameters.mat', 'params');
        
        % 在命令窗口显示参数
        fprintf('\n=== 保存的波形参数 ===\n');
        fprintf('d1 (峰值前): %d 点\n', current_d1);
        fprintf('d2 (峰值后): %d 点\n', current_d2);
        fprintf('总长度: %d 点\n', current_d1 + current_d2 + 1);
        fprintf('测试数据: H = %d\n', current_H);
        fprintf('参数已保存到: waveform_parameters.mat\n');
        fprintf('=====================\n\n');
        
        set(info_text, 'String', sprintf('参数已保存!\nd1=%d, d2=%d\n总长度=%d点', ...
            current_d1, current_d2, current_d1+current_d2+1));
    end
    
    % 设置回调函数
    set(update_btn, 'Callback', @(~,~) update_display());
    set(reload_btn, 'Callback', @(~,~) reload_data());
    set(save_params_btn, 'Callback', @(~,~) save_parameters());
    
    % 初始显示
    update_display();
    
    fprintf('波形查看工具已启动!\n');
    fprintf('- 可以修改H值来查看不同的数据文件\n');
    fprintf('- 调整d1和d2值来设置波形提取范围\n');
    fprintf('- 修改峰值编号来查看不同的波形\n');
    fprintf('- 点击"保存参数"将参数保存到文件\n\n');
end