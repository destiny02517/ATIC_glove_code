%% Cut cycles by manual templates (2-3 cycles) + template matching + colored check
clear; clc; close all;

%% ---------------- User settings ----------------
h = 166;  % <<< 改成你的 scope 编号（例如 85~93 等）
file1 = sprintf("scope_%d_1.csv", h);   % Ch1
file2 = sprintf("scope_%d_2.csv", h);   % Ch2

% 你想保存的编号（保持和你之前一致：H+1）
H = h - 108;   % 保存时用 H+1

% 手动选模板段数：建议 2~3
Kmax = 3;            % 最多选 3 段
Kmin = 2;            % 至少选 2 段（没选够会提示你继续选）

% 模板周期统一长度（重采样后每周期点数）
% 你采样率 1~2kHz，25bpm 周期约 2.4s -> 2400~4800点
% 这里用 3000/4000 都可以；也可以设成 2500、3500 等
cycleLen = 2000;     % <<< 每周期最终长度（用于导出和可视化对齐）

% 轻微平滑（秒）
smoothWin_sec = 0.008;   % 6~15ms 都行，太大可能抹平边沿

% findpeaks 参数（自动估计周期用）
minDistFactor = 0.65;    % 最小周期距离 = factor * T_est
prominenceFactor = 0.6;  % 峰显著性阈值 = factor * std(score)

% 打分融合（Ch1/Ch2）
w1 = 0.7;  % Ch1 权重
w2 = 0.3;  % Ch2 权重（如果 Ch2 很干净，可提高到 0.5）

% 保存文件名
outV1 = sprintf("V1_%d.mat", H+1);
outV2 = sprintf("V2_%d.mat", H+1);

%% ---------------- Load 2 channels ----------------
[t, ch1] = load_1ch(file1);
[t2, ch2] = load_1ch(file2);

n = min(numel(t), numel(t2));
t = t(1:n); ch1 = ch1(1:n); ch2 = ch2(1:n);

dt = median(diff(t));
fs = 1/dt;
fprintf("Loaded %d samples, fs ~ %.2f Hz\n", n, fs);

%% ---------------- Preprocess ----------------
smoothWin = max(3, round(smoothWin_sec*fs));
ch1s = movmean(ch1, smoothWin);
ch2s = movmean(ch2, smoothWin);

%% ---------------- Manual pick templates with Zoom button (GUI) ----------------
[tmpl1_all, tmpl2_all, dur_all, fs] = pick_templates_ui(t, ch1s, ch2s, fs, Kmin, Kmax, cycleLen);

K = size(tmpl1_all,1);
fprintf("Picked %d template cycles.\n", K);

% 模板取平均（更稳）
tmpl1 = mean(tmpl1_all, 1);
tmpl2 = mean(tmpl2_all, 1);

% 周期估计：用你手选段的中位数
T_est = median(dur_all);
fprintf("Estimated period from templates: T_est ~ %.3f s\n", T_est);


%% ---------------- Template matching (NCC) on full signal ----------------
% 先把整段也做 zscore（整体趋势会影响相关）
x1 = zscore0(ch1s(:));
x2 = zscore0(ch2s(:));
tmpl1n = zscore0(tmpl1(:));
tmpl2n = zscore0(tmpl2(:));

score1 = ncc_score(x1, tmpl1n);
score2 = ncc_score(x2, tmpl2n);

% 融合得分
score = w1*score1 + w2*score2;

% score 对应的起点索引：1..(n-cycleLen+1)
t_score = t(1:(n-cycleLen+1));

%% ---------------- Find all cycles from score peaks ----------------
minDist = max(1, round(minDistFactor*T_est*fs));
minProm = max(0.05, prominenceFactor*std(score));

[pk, loc] = findpeaks(score, ...
    "MinPeakDistance", minDist, ...
    "MinPeakProminence", minProm);

starts = loc;  % 每个 peak 表示“最像模板”的窗口起点
numCycles = numel(starts);

fprintf("Detected %d cycles (peaks on score).\n", numCycles);
if numCycles < 2
    warning("检测到的周期太少：可尝试降低 prominenceFactor 或 minDistFactor，或重新选更典型的模板段。");
end

%% ---------------- Build fixed-length cycle segments (Ch1+Ch2) ----------------
X11 = zeros(0, cycleLen);
X22 = zeros(0, cycleLen);
starts_kept = [];

for i = 1:numCycles
    s = starts(i);
    e = s + cycleLen - 1;
    if e > n
        continue;
    end

    % 防止重叠：如果下一个 starts 太近，可以跳过或保留（这里保留但你也可加规则）
    X11(end+1,:) = ch1(s:e).';
    X22(end+1,:) = ch2(s:e).';
    starts_kept(end+1) = s;
end

numKept = size(X11,1);
fprintf("Kept %d fixed-length cycles for export.\n", numKept);


% 保存 MAT
save(outV1, "X11");
save(outV2, "X22");
fprintf("Saved: %s and %s\n", outV1, outV2);

%% ---------------- Visualization 1: score + peaks ----------------
figure("Name","Template match score + peaks");
plot(t_score, score, "LineWidth", 1); grid on; hold on;
plot(t_score(loc), pk, "rv", "MarkerFaceColor","r");
xlabel("t (s)"); ylabel("score (NCC fused)");
title("Template matching score (fused) and detected cycle starts");

%% ---------------- Visualization 2: colored cycles on raw waveform ----------------
figure("Name","Colored cycles on Ch1"); hold on;
plot(t, ch1, "Color", [0.7 0.7 0.7]); % 灰色背景原始波形
grid on; xlabel("t (s)"); ylabel("Ch1 (V)");
title("Ch1: each detected cycle colored (check segmentation)");

C = lines(max(1,numKept));  % 自动生成颜色
for i = 1:numKept
    s = starts_kept(i);
    e = s + cycleLen - 1;
    plot(t(s:e), ch1(s:e), "LineWidth", 1.5, "Color", C(mod(i-1,size(C,1))+1,:));
end
plot(t(starts_kept), ch1(starts_kept), "kv", "MarkerFaceColor","y"); % 起点标记
legend("raw (gray)", "cycles (colored)", "cycle starts", "Location","best");

%% ---------------- Visualization 3: overlay cycles (aligned) ----------------
% 叠加对齐检查（用 cycleLen 点的相对时间轴）
t_rel = (0:cycleLen-1)/fs;

figure("Name","Overlay cycles Ch1"); hold on; grid on;
for i = 1:numKept
    plot(t_rel, X11(i,:), "LineWidth", 0.6);
end
plot(t_rel, mean(X11,1), "k", "LineWidth", 3);
xlabel("time within cycle (s)"); ylabel("Ch1 (V)");
title(sprintf("Overlay Ch1 cycles: %d cycles + mean", numKept));

figure("Name","Overlay cycles Ch2"); hold on; grid on;
for i = 1:numKept
    plot(t_rel, X22(i,:), "LineWidth", 0.6);
end
plot(t_rel, mean(X22,1), "k", "LineWidth", 3);
xlabel("time within cycle (s)"); ylabel("Ch2 (V)");
title(sprintf("Overlay Ch2 cycles: %d cycles + mean", numKept));

%% ================= Local functions =================
function [t, y] = load_1ch(fname)
    A = readmatrix(fname);
    if size(A,2) >= 2
        t = A(:,1);
        y = A(:,2);
    else
        y = A(:,1);
        t = (0:numel(y)-1).';
    end
    m = ~(isnan(t) | isnan(y));
    t = t(m); y = y(m);
end

function y = zscore0(x)
    x = x(:);
    y = x - mean(x);
    y = y / (std(y) + 1e-12);
end

function xr = resample_to_len(x, L)
    x = x(:);
    if numel(x) == L
        xr = x;
        return;
    end
    u  = linspace(0,1,numel(x));
    uu = linspace(0,1,L);
    xr = interp1(u, x, uu, "linear", "extrap");
end

function score = ncc_score(x, tmpl)
    % Normalized cross-correlation score between x and tmpl (both column)
    x = x(:);
    tmpl = tmpl(:);
    L = numel(tmpl);
    n = numel(x);
    if n < L
        error("Signal shorter than template.");
    end

    % numerator: dot(window, tmpl)
    num = conv(x, flipud(tmpl), 'valid');  % length n-L+1

    % denominator: ||window|| * ||tmpl||
    winE = movsum(x.^2, [L-1 0]);   % length n
    winE = winE(L:end);            % align to 'valid'
    den = sqrt(winE) * sqrt(sum(tmpl.^2));

    score = num ./ (den + 1e-12);
end

function [tmpl1_all, tmpl2_all, dur_all, fs] = pick_templates_ui(t, ch1s, ch2s, fs, Kmin, Kmax, cycleLen)
    % GUI: Zoom / Pan / Select ROI / Done
    % ROI: draw rectangle on plot, use x-range as one cycle
    
    tmpl1_all = [];
    tmpl2_all = [];
    dur_all   = [];
    k = 0;

    fig = figure("Name","Pick templates (Zoom then box-select)", "NumberTitle","off");
    ax = axes(fig);
    plot(ax, t, ch1s, "LineWidth", 1); grid(ax,"on");
    xlabel(ax,"t (s)"); ylabel(ax,"Ch1 (V)");
    title(ax, sprintf("先点 Zoom/Pan 放大，然后点『框选周期』拖框选一段周期 (选 %d~%d 段)，最后点 Done", Kmin, Kmax));

    % ---- Buttons ----
    btnW = 110; btnH = 28; gap = 10;
    left = 15; top  = 15;

    % Zoom toggle
    hZoom = uicontrol(fig, "Style","togglebutton", "String","Zoom OFF", ...
        "Position",[left, top, btnW, btnH], ...
        "Callback", @(src,evt)cbZoom(src, fig));

    % Pan toggle
    hPan = uicontrol(fig, "Style","togglebutton", "String","Pan OFF", ...
        "Position",[left+btnW+gap, top, btnW, btnH], ...
        "Callback", @(src,evt)cbPan(src, fig, hZoom));

    % Select ROI
    hSel = uicontrol(fig, "Style","pushbutton", "String","框选周期", ...
        "Position",[left+2*(btnW+gap), top, btnW, btnH], ...
        "Callback", @(src,evt)cbSelect(fig));

    % Reset view
    hReset = uicontrol(fig, "Style","pushbutton", "String","Reset", ...
        "Position",[left+3*(btnW+gap), top, btnW, btnH], ...
        "Callback", @(src,evt)cbReset(ax, t));

    % Done
    hDone = uicontrol(fig, "Style","pushbutton", "String","Done", ...
        "Position",[left+4*(btnW+gap), top, btnW, btnH], ...
        "Callback", @(src,evt)cbDone(fig));

    % info text
    hInfo = uicontrol(fig, "Style","text", "String","状态：请先 Zoom/Pan 后框选", ...
        "HorizontalAlignment","left", ...
        "Position",[left, top+btnH+6, 700, 20]);

    % appdata flags
    setappdata(fig, "action", "");    % "select" or "done"
    setappdata(fig, "selected", []);  % [t0 t1]

    % ---- main loop ----
    while isvalid(fig) && (k < Kmax)
        setappdata(fig, "action", "");
        uiwait(fig);  % wait for button callbacks to call uiresume

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
            % turn off zoom/pan before selecting
            try
                zoom(fig, "off"); pan(fig, "off");
            catch
            end
            set(hZoom, "Value", 0); set(hZoom, "String","Zoom OFF");
            set(hPan,  "Value", 0); set(hPan,  "String","Pan OFF");

            set(hInfo, "String", "状态：请用鼠标拖一个矩形框选一个周期（只取x范围）...");
            drawnow;

            % draw ROI rectangle
            r = drawrectangle(ax, "Color","r", "LineWidth", 1.5);
            wait(r);  % wait user finish dragging / double click / press Enter

            pos = r.Position; % [x y w h]
            delete(r);

            t0 = pos(1);
            t1 = pos(1) + pos(3);

            if ~(isfinite(t0) && isfinite(t1)) || abs(t1-t0) < 1e-6
                set(hInfo, "String", "状态：框选无效，请重试。");
                continue;
            end
            if t1 < t0, tmp=t0; t0=t1; t1=tmp; end

            s0 = find(t >= t0, 1, "first");
            e0 = find(t <= t1, 1, "last");

            if isempty(s0) || isempty(e0) || e0 <= s0+10
                set(hInfo, "String", "状态：框选区间太短或超出范围，请重试。");
                continue;
            end

            seg1 = ch1s(s0:e0);
            seg2 = ch2s(s0:e0);
            dur  = t(e0) - t(s0);

            % resample to fixed length
            seg1r = resample_to_len(seg1, cycleLen);
            seg2r = resample_to_len(seg2, cycleLen);

            % z-score normalize
            seg1r = zscore0(seg1r);
            seg2r = zscore0(seg2r);

            tmpl1_all = [tmpl1_all; seg1r(:)'];
            tmpl2_all = [tmpl2_all; seg2r(:)'];
            dur_all(end+1) = dur;

            k = k + 1;
            hold(ax, "on");
            plot(ax, t(s0:e0), ch1s(s0:e0), "LineWidth", 2);
            text(ax, t(s0), ch1s(s0), sprintf("  T%d", k), "FontSize", 12, "FontWeight","bold");
            hold(ax, "off");

            set(hInfo, "String", sprintf("状态：已选 %d 段模板。可继续 Zoom/Pan，再点『框选周期』；或点 Done 结束。", k));
        end
    end

    if isvalid(fig)
        close(fig);
    end

    % -------- nested callbacks --------
    function cbZoom(src, f)
        if src.Value == 1
            zoom(f, "on");
            src.String = "Zoom ON";
        else
            zoom(f, "off");
            src.String = "Zoom OFF";
        end
    end

    function cbPan(src, f, hZoomLocal)
        % 如果 Pan ON，就把 Zoom 关掉，避免冲突
        if src.Value == 1
            try
                zoom(f, "off");
            catch
            end
            if isvalid(hZoomLocal)
                hZoomLocal.Value = 0;
                hZoomLocal.String = "Zoom OFF";
            end
            pan(f, "on");
            src.String = "Pan ON";
        else
            pan(f, "off");
            src.String = "Pan OFF";
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
        try
            xlim(axLocal, [tLocal(1), tLocal(end)]);
        catch
        end
    end
end



