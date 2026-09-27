%% Cut cycles from ONE CSV with 40 channels (I0~I39, no time)
% Manual templates (2-3 cycles) + NCC template matching + colored check
% + Export each cycle to separate CSV files in a new folder
clear; clc; close all;

%% ---------------- User settings ----------------
csvFile = "103.csv";     % <<< 改成你的文件名（第一行表头 I0~I39）
fs = 500;               % <<< 你的采样率(Hz)，没有时间轴必须提供

% 手动选模板段数
Kmax = 3;
Kmin = 2;

% 每周期最终长度（点数）
cycleLen = 1300;        % <<< 固定长度输出（点数）

% 轻微平滑（秒）
smoothWin_sec = 0.008;

% findpeaks 参数（自动估计周期用）
minDistFactor     = 0.65;     % MinPeakDistance = factor*T_est*fs
prominenceFactor  = 0.6;      % MinPeakProminence = factor*std(score)

% -------- 关键：多通道策略 --------
refCh    = 0;                  % <<< 只在这个通道上框选模板（0~39，对应 I0~I39）
matchChs = [0 5 10 15];        % <<< 用这些通道做NCC并融合（可只写 [refCh]）
w = [];                        % <<< 融合权重（留空=均分；或写成与 matchChs 等长）

% 输出 MAT
outMat = "Cycles_40ch.mat";

% -------- 导出 CSV 设置 --------
exportCyclesToCSV = true;      % <<< 是否导出每周期一个CSV
exportTimeAxis    = true;      % <<< CSV 第一列是否加 t（秒）
csvFolderPrefix   = "CyclesCSV"; % 输出文件夹前缀：CyclesCSV_129

%% ---------------- Load 40 channels ----------------
Xraw = readmatrix(csvFile);        % 多数情况下会把表头读成 NaN 行
Xraw = Xraw(:, 1:40);              % 只取前40列（如果你csv还有别的列）

% 去掉包含 NaN 的行（把表头那一行清掉）
m = all(~isnan(Xraw), 2);
Xraw = Xraw(m, :);

% 检查维度
assert(size(Xraw,2) >= 40, "CSV 列数不足 40：请确认文件是否为 I0~I39");
Xraw = Xraw(:,1:40);

N = size(Xraw,1);
t = (0:N-1)'/fs;
fprintf("Loaded %d samples, %d channels, fs = %.2f Hz\n", N, size(Xraw,2), fs);

%% ---------------- Preprocess (smooth) ----------------
smoothWin = max(3, round(smoothWin_sec*fs));
Xs = movmean(Xraw, smoothWin, 1);    % 对每一列做平滑

%% ---------------- Manual pick templates on ref channel (GUI) ----------------
[yT_all, dur_all] = pick_templates_ui_1ch(t, Xs(:,refCh+1), Kmin, Kmax, cycleLen, fs);
K = size(yT_all,1);
fprintf("Picked %d template cycles on I%d.\n", K, refCh);

tmpl = mean(yT_all, 1);              % 模板取平均
T_est = median(dur_all);             % 用手选周期时长估计周期
fprintf("Estimated period from templates: T_est ~ %.3f s\n", T_est);

%% ---------------- Template matching (NCC) on full signal ----------------
if isempty(w)
    w = ones(1, numel(matchChs)) / numel(matchChs);
else
    w = w(:)' / sum(w);
end

tmpln = zscore0(tmpl(:));
L = cycleLen;

score = zeros(N-L+1, 1);
for i = 1:numel(matchChs)
    ch = matchChs(i);
    x  = zscore0(Xs(:, ch+1));           % 整段 zscore
    s  = ncc_score(x, tmpln);            % NCC score
    score = score + w(i) * s(:);
end
t_score = t(1:(N-L+1));

%% ---------------- Find all cycles from score peaks ----------------
minDist = max(1, round(minDistFactor*T_est*fs));
minProm = max(0.05, prominenceFactor*std(score));

[pk, loc] = findpeaks(score, ...
    "MinPeakDistance", minDist, ...
    "MinPeakProminence", minProm);

starts = loc;   % 每个峰对应一个窗口起点
numCycles = numel(starts);
fprintf("Detected %d cycles (peaks on fused score).\n", numCycles);

if numCycles < 2
    warning("检测到的周期太少：尝试降低 prominenceFactor/minDistFactor 或重新选更典型模板。");
end

%% ---------------- Build fixed-length cycle segments for ALL 40 channels ----------------
starts_kept = [];
X = zeros(0, cycleLen, 40);

for i = 1:numCycles
    s = starts(i);
    e = s + cycleLen - 1;
    if e > N
        continue;
    end

    seg40 = Xraw(s:e, :);                    % cycleLen x 40
    X(end+1, :, :) = permute(seg40, [3 1 2]);% 1 x L x 40
    starts_kept(end+1) = s;
end

numKept = size(X,1);
fprintf("Kept %d fixed-length cycles for export.\n", numKept);

%% ---------------- Save MAT ----------------
meta.csvFile = csvFile;
meta.fs = fs;
meta.cycleLen = cycleLen;
meta.refCh = refCh;
meta.matchChs = matchChs;
meta.weights = w;
meta.T_est = T_est;

save(outMat, "X", "starts_kept", "meta", "-v7.3");
fprintf("Saved: %s\n", outMat);

%% ---------------- Export each cycle to separate CSV ----------------
if exportCyclesToCSV && numKept > 0
    % --- 根据输入CSV文件名生成文件夹名（不含扩展名）---
[~, baseName, ~] = fileparts(csvFile);                 % 例如 "82"
% outDir = fullfile(pwd, sprintf("%s_cycles_%d", baseName, numKept));
outDir = fullfile(pwd, sprintf("%s_cycles", baseName));

if ~exist(outDir, "dir")
    mkdir(outDir);
end

    % CSV 表头：I0~I39
    C = size(X,3); % should be 40
    hdr = strings(1, C);
    for k = 0:C-1
        hdr(k+1) = "I" + k;
    end

    % 可选时间轴
    if exportTimeAxis
        t_rel = (0:cycleLen-1)'/fs;   % 秒
    end

    % 逐周期写文件
    for i = 1:numKept
        Xi = squeeze(X(i,:,:));  % cycleLen x 40

        if exportTimeAxis
            M = [t_rel, Xi];                 % cycleLen x 41
            header = ["t", hdr];             % 1 x 41
        else
            M = Xi;                          % cycleLen x 40
            header = hdr;                    % 1 x 40
        end

        % 文件名：cycle_001.csv（可加起点：_start_000123）
%        outFile = fullfile(outDir, sprintf("cycle_%03d_start_%06d.csv", i, starts_kept(i)));
         outFile = fullfile(outDir, sprintf("cycle_%d.csv", i));

        % 写入：先写表头，再追加数据
        writecell(cellstr(header), outFile);
        writematrix(M, outFile, "WriteMode","append");
    end

    fprintf("Exported %d cycle CSV files to folder:\n%s\n", numKept, outDir);
end

%% ---------------- Visualization 1: score + peaks ----------------
figure("Name","Template match score + peaks");
plot(t_score, score, "LineWidth", 1); grid on; hold on;
plot(t_score(loc), pk, "rv", "MarkerFaceColor","r");
xlabel("t (s)"); ylabel("score (NCC fused)");
title("Template matching score (fused) and detected cycle starts");

%% ---------------- Visualization 2: colored cycles on ref channel ----------------
ref = refCh + 1;
figure("Name",sprintf("Colored cycles on I%d", refCh)); hold on;
plot(t, Xraw(:,ref), "Color", [0.7 0.7 0.7]); grid on;
xlabel("t (s)"); ylabel(sprintf("I%d", refCh));
title(sprintf("I%d: detected cycles colored (check segmentation)", refCh));

Cmap = lines(max(1,numKept));
for i = 1:numKept
    s = starts_kept(i); e = s + cycleLen - 1;
    plot(t(s:e), Xraw(s:e,ref), "LineWidth", 1.5, "Color", Cmap(mod(i-1,size(Cmap,1))+1,:));
end
plot(t(starts_kept), Xraw(starts_kept,ref), "kv", "MarkerFaceColor","y");
legend("raw (gray)", "cycles (colored)", "cycle starts", "Location","best");

%% ---------------- Visualization 3: overlay cycles (aligned) on a few channels ----------------
t_rel_plot = (0:cycleLen-1)/fs;
showChs = unique([refCh, matchChs]);
showChs = showChs(1:min(numel(showChs), 4));   % 最多展示 4 个通道，避免太多图

for ii = 1:numel(showChs)
    ch = showChs(ii);
    figure("Name", sprintf("Overlay cycles I%d", ch)); hold on; grid on;
    Y = squeeze(X(:,:,ch+1)); % numKept x L
    for i = 1:numKept
        plot(t_rel_plot, Y(i,:), "LineWidth", 0.6);
    end
    plot(t_rel_plot, mean(Y,1), "k", "LineWidth", 3);
    xlabel("time within cycle (s)");
    ylabel(sprintf("I%d", ch));
    title(sprintf("Overlay I%d cycles: %d cycles + mean", ch, numKept));
end

%% ================= Local functions =================
function y = zscore0(x)
    x = x(:);
    y = x - mean(x);
    y = y / (std(y) + 1e-12);
end

function xr = resample_to_len(x, L)
    x = x(:);
    if numel(x) == L, xr = x; return; end
    u  = linspace(0,1,numel(x));
    uu = linspace(0,1,L);
    xr = interp1(u, x, uu, "linear", "extrap");
end

function score = ncc_score(x, tmpl)
    % Normalized cross-correlation score between x and tmpl (both column)
    x = x(:); tmpl = tmpl(:);
    L = numel(tmpl); n = numel(x);
    if n < L, error("Signal shorter than template."); end

    num = conv(x, flipud(tmpl), 'valid');      % dot(window, tmpl), length n-L+1
    winE = movsum(x.^2, [L-1 0]);              % length n
    winE = winE(L:end);                        % align to 'valid'
    den  = sqrt(winE) * sqrt(sum(tmpl.^2));

    score = num ./ (den + 1e-12);
end

function [tmpl_all, dur_all] = pick_templates_ui_1ch(t, y, Kmin, Kmax, cycleLen, fs)
    % GUI: Zoom / Pan / Select ROI / Done (select on ONE channel)
    tmpl_all = [];
    dur_all  = [];
    k = 0;

    fig = figure("Name","Pick templates (Zoom then box-select)", "NumberTitle","off");
    ax = axes(fig);
    plot(ax, t, y, "LineWidth", 1); grid(ax,"on");
    xlabel(ax,"t (s)"); ylabel(ax,"signal");
    title(ax, sprintf("先点 Zoom/Pan 放大，然后点『框选周期』拖框选一段周期 (选 %d~%d 段)，最后点 Done", Kmin, Kmax));

    btnW = 110; btnH = 28; gap = 10;
    left = 15; top  = 15;

    hZoom = uicontrol(fig, "Style","togglebutton", "String","Zoom OFF", ...
        "Position",[left, top, btnW, btnH], ...
        "Callback", @(src,evt)cbZoom(src, fig));

    hPan = uicontrol(fig, "Style","togglebutton", "String","Pan OFF", ...
        "Position",[left+btnW+gap, top, btnW, btnH], ...
        "Callback", @(src,evt)cbPan(src, fig, hZoom));

    uicontrol(fig, "Style","pushbutton", "String","框选周期", ...
        "Position",[left+2*(btnW+gap), top, btnW, btnH], ...
        "Callback", @(src,evt)cbSelect(fig));

    uicontrol(fig, "Style","pushbutton", "String","Reset", ...
        "Position",[left+3*(btnW+gap), top, btnW, btnH], ...
        "Callback", @(src,evt)cbReset(ax, t));

    uicontrol(fig, "Style","pushbutton", "String","Done", ...
        "Position",[left+4*(btnW+gap), top, btnW, btnH], ...
        "Callback", @(src,evt)cbDone(fig));

    hInfo = uicontrol(fig, "Style","text", "String","状态：请先 Zoom/Pan 后框选", ...
        "HorizontalAlignment","left", ...
        "Position",[left, top+btnH+6, 700, 20]);

    setappdata(fig, "action", "");

    while isvalid(fig) && (k < Kmax)
        setappdata(fig, "action", "");
        uiwait(fig);

        if ~isvalid(fig), break; end
        act = getappdata(fig, "action");

        if strcmp(act, "done")
            if k < Kmin
                set(hInfo, "String", sprintf("状态：至少选 %d 段模板，目前只有 %d 段，请继续框选。", Kmin, k));
                continue;
            else
                break;
            end
        elseif strcmp(act, "select")
            try, zoom(fig,"off"); pan(fig,"off"); end %#ok<TRYNC>
            set(hZoom, "Value", 0); set(hZoom, "String","Zoom OFF");
            set(hPan,  "Value", 0); set(hPan,  "String","Pan OFF");

            set(hInfo, "String", "状态：请用鼠标拖一个矩形框选一个周期（只取x范围）...");
            drawnow;

            r = drawrectangle(ax, "Color","r", "LineWidth", 1.5);
            wait(r);

            pos = r.Position; delete(r);
            t0 = pos(1); t1 = pos(1)+pos(3);
            if t1 < t0, tmp=t0; t0=t1; t1=tmp; end

            s0 = find(t >= t0, 1, "first");
            e0 = find(t <= t1, 1, "last");

            if isempty(s0) || isempty(e0) || e0 <= s0+10
                set(hInfo, "String", "状态：框选区间太短或超出范围，请重试。");
                continue;
            end

            seg = y(s0:e0);
            dur = (e0 - s0)/fs;   % 用点数更稳（因为 t 是构造的）

            segR = resample_to_len(seg, cycleLen);
            segR = zscore0(segR);

            tmpl_all = [tmpl_all; segR(:)'];
            dur_all(end+1) = dur;

            k = k + 1;
            hold(ax, "on");
            plot(ax, t(s0:e0), y(s0:e0), "LineWidth", 2);
            text(ax, t(s0), y(s0), sprintf("  T%d", k), "FontSize", 12, "FontWeight","bold");
            hold(ax, "off");

            set(hInfo, "String", sprintf("状态：已选 %d 段模板。可继续 Zoom/Pan，再点『框选周期』；或点 Done 结束。", k));
        end
    end

    if isvalid(fig), close(fig); end
end

function cbZoom(src, f)
    if src.Value == 1
        zoom(f, "on");  src.String = "Zoom ON";
    else
        zoom(f, "off"); src.String = "Zoom OFF";
    end
end

function cbPan(src, f, hZoomLocal)
    if src.Value == 1
        try, zoom(f, "off"); end %#ok<TRYNC>
        if isvalid(hZoomLocal)
            hZoomLocal.Value = 0; hZoomLocal.String = "Zoom OFF";
        end
        pan(f, "on");  src.String = "Pan ON";
    else
        pan(f, "off"); src.String = "Pan OFF";
    end
end

function cbSelect(f)
    setappdata(f, "action", "select");
    uiresume(f);
end

function cbDone(f)
    setappdata(f, "action", "done");
    uiresume(f);
end

function cbReset(axLocal, tLocal)
    try, xlim(axLocal, [tLocal(1), tLocal(end)]); end %#ok<TRYNC>
end
