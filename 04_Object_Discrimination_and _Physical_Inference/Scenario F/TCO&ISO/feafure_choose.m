%% Cut cycles by manual templates (2-3 cycles) + template matching
%  NEW LOGIC:
%  - segment cycles by adjacent detected starts (variable length)
%  - show colored cycles
%  - click to select 50 cycles
%  - compute feature waveform (mean of selected cycles after resample) and save to CSV

clear; clc; close all;

%% ---------------- User settings ----------------
h = 103;  % <<< scope 编号
file1 = sprintf("scope_%d_1.csv", h);   % Ch1
file2 = sprintf("scope_%d_2.csv", h);   % Ch2

H = h - 82;   % 只是沿用你之前编号习惯

% 手动选模板段数
Kmax = 3;
Kmin = 2;

% 用于模板匹配的窗口长度（模板会重采样到这个长度）
cycleLen = 3000;     % <<< 仍然用于模板匹配/特征统一长度

% 特征波形长度（你最终保存的特征波形点数）
Lfeat = cycleLen;    % <<< 一般就等于 cycleLen；想更短可设 1500 等

% 平滑
smoothWin_sec = 0.008;

% findpeaks 参数（自动估计周期用）
minDistFactor = 0.65;
prominenceFactor = 0.6;

% 打分融合（Ch1/Ch2）
w1 = 0.7;
w2 = 0.3;

% 周期长度合法范围（用 T_est 限制异常段）
lenMinFactor = 0.50;   % 最短 >= 0.50*T_est
lenMaxFactor = 1.60;   % 最长 <= 1.60*T_est

% 你要选多少个周期来算特征波形
Npick = 50;

% 输出 CSV 文件名（特征波形）
outCSV = sprintf("FeatureWaveform_H%d.csv", H+1);

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

%% ---------------- Manual pick templates (GUI) ----------------
[tmpl1_all, tmpl2_all, dur_all, fs] = pick_templates_ui(t, ch1s, ch2s, fs, Kmin, Kmax, cycleLen);

K = size(tmpl1_all,1);
fprintf("Picked %d template cycles.\n", K);

tmpl1 = mean(tmpl1_all, 1);
tmpl2 = mean(tmpl2_all, 1);

T_est = median(dur_all);
fprintf("Estimated period from templates: T_est ~ %.3f s\n", T_est);

%% ---------------- Template matching (NCC) on full signal ----------------
x1 = zscore0(ch1s(:));
x2 = zscore0(ch2s(:));
tmpl1n = zscore0(tmpl1(:));
tmpl2n = zscore0(tmpl2(:));

score1 = ncc_score(x1, tmpl1n);
score2 = ncc_score(x2, tmpl2n);

score = w1*score1 + w2*score2;
t_score = t(1:(n-cycleLen+1));  % score 对应的时间轴

%% ---------------- Find all cycle starts from score peaks ----------------
minDist = max(1, round(minDistFactor*T_est*fs));
minProm = max(0.05, prominenceFactor*std(score));

[pk, loc] = findpeaks(score, ...
    "MinPeakDistance", minDist, ...
    "MinPeakProminence", minProm);

starts = loc(:);         % score 索引（也是原信号起点索引，因为 score 是 valid 卷积）
pks    = pk(:);
numStarts = numel(starts);

fprintf("Detected %d starts (peaks on score).\n", numStarts);
if numStarts < 2
    warning("起点太少：可尝试降低 prominenceFactor 或 minDistFactor，或重新选更典型模板段。");
end

%% ---------------- Build variable-length cycles by adjacent starts ----------------
% cycle i: [starts(i), starts(i+1)-1]
% last cycle: [starts(end), starts(end)+round(T_est*fs)-1]
cycles = struct("s",{}, "e",{}, "len",{}, "score",{}, "t0",{}, "dur",{});
if numStarts >= 1
    Lmin = round(lenMinFactor * T_est * fs);
    Lmax = round(lenMaxFactor * T_est * fs);

    for i = 1:numStarts
        s = starts(i);

        if i < numStarts
            e = starts(i+1) - 1;
        else
            e = s + round(T_est*fs) - 1;
        end

        e = min(e, n);
        if e <= s+10, continue; end

        L = e - s + 1;
        if L < Lmin || L > Lmax
            % 过滤异常长短周期（你也可以注释掉这段不过滤）
            continue;
        end

        c.s = s;
        c.e = e;
        c.len = L;
        c.score = score(s);            % 起点处得分
        c.t0 = t(s);
        c.dur = t(e) - t(s);
        cycles(end+1) = c; %#ok<SAGROW>
    end
end
numCycles = numel(cycles);
fprintf("Kept %d cycles after length filter.\n", numCycles);

if numCycles < Npick
    warning("可用周期只有 %d 个，少于你要选的 %d 个。之后会用全部可用周期算特征。", numCycles, Npick);
end

%% ---------------- Visualization: score + peaks ----------------
figure("Name","Template match score + peaks");
plot(t_score, score, "LineWidth", 1); grid on; hold on;
plot(t_score(loc), pk, "rv", "MarkerFaceColor","r");
xlabel("t (s)"); ylabel("score (NCC fused)");
title("Template matching score (fused) and detected cycle starts");

%% ---------------- Click-to-select UI on colored cycles ----------------
[selectedIdx, autoIdx] = select_cycles_click_ui(t, ch1, cycles, Npick);

% selectedIdx: indices into cycles struct
if isempty(selectedIdx)
    warning("你没有选择任何周期，将使用 AutoTop50（或全部可用）来生成特征波形。");
    selectedIdx = autoIdx;
end

% 如果用户选多了，取其中 score 最高的 Npick 个
if numel(selectedIdx) > Npick
    sc = arrayfun(@(k)cycles(k).score, selectedIdx);
    [~, ord] = sort(sc, "descend");
    selectedIdx = selectedIdx(ord(1:Npick));
end

% 如果用户选少了，补齐（按 score 从高到低补）
if numel(selectedIdx) < min(Npick, numCycles)
    need = Npick - numel(selectedIdx);
    allIdx = 1:numCycles;
    rest = setdiff(allIdx, selectedIdx, "stable");
    sc2 = arrayfun(@(k)cycles(k).score, rest);
    [~, ord2] = sort(sc2, "descend");
    add = rest(ord2(1:min(need, numel(rest))));
    selectedIdx = [selectedIdx(:); add(:)];
end

fprintf("Final picked cycles: %d\n", numel(selectedIdx));

%% ---------------- Compute feature waveform (mean of selected cycles) ----------------
% 取原始 ch1/ch2（不 zscore），每个周期重采样到 Lfeat，求均值
S1 = zeros(numel(selectedIdx), Lfeat);
S2 = zeros(numel(selectedIdx), Lfeat);

for ii = 1:numel(selectedIdx)
    k = selectedIdx(ii);
    s = cycles(k).s; e = cycles(k).e;

    seg1 = ch1(s:e);
    seg2 = ch2(s:e);

%     % 可选：去掉每段的 DC（更强调形状；不想去 DC 就注释掉）
%     seg1 = seg1 - mean(seg1);
%     seg2 = seg2 - mean(seg2);

    S1(ii,:) = resample_to_len(seg1, Lfeat);
    S2(ii,:) = resample_to_len(seg2, Lfeat);
end
%% ---- Spike alignment (micro-shift) ----
%% ---- Peak-preserving alignment + feature (replace old NCC micro-shift) ----
% 自动找尖峰最活跃窗口（基于平均 |diff| 能量）
idxWin = auto_spike_window(S2, round(0.18*Lfeat));   % 0.18 可调 0.12~0.25

% 用“导数峰(最大斜率)”分别对齐 Ch1/Ch2（不要再用 Ch1 lag 去对齐 Ch2）
maxLag = round(0.10*Lfeat);                          % 允许 +/-10% 平移（可调 0.05~0.15）
[S1a, info1] = align_by_spike(S1, idxWin, maxLag);
[S2a, info2] = align_by_spike(S2, idxWin, maxLag);

S1 = S1a;
S2 = S2a;

% 只用尖峰方向一致 + 尖峰强的周期来算特征（避免互相抵消）
sg = info2.sign;                     % +1 上跳，-1 下跳
domSign = mode_sign(sg);             % 数量最多的一类

strength = info2.strength;           % 尖峰强度（|导数峰|）
thr = prctile(strength, 40);         % 只用强度前 60%（可调 20~70）
keep = (sg == domSign) & (strength >= thr);

% 兜底：至少保留 15 条
if nnz(keep) < 20
    [~,ord] = sort(strength, 'descend');
    keep = false(size(keep));
    keep(ord(1:min(15,numel(ord)))) = true;
end

fprintf("Feature uses %d/%d cycles (dominant sign=%+d)\n", nnz(keep), numel(keep), domSign);

% 加权平均（更保尖峰）
w = strength(keep);
feat1 = weighted_mean(S1(keep,:), w);
feat2 = weighted_mean(S2(keep,:), w);

% 重采样后的时间轴：用 T_est 更合理（别用 fs 推）
t_rel = linspace(0, T_est, Lfeat);


%% ---------------- Plot overlay + feature waveform ----------------
figure("Name","Selected cycles overlay + feature (Ch1)");
hold on; grid on;
for ii = 1:size(S1,1)
    plot(t_rel, S1(ii,:), "LineWidth", 0.6);
end
plot(t_rel, feat1, "k", "LineWidth", 3);
xlabel("time within cycle (s)"); ylabel("Ch1 (V, DC removed)");
title(sprintf("Ch1: %d selected cycles overlay + mean feature", size(S1,1)));

figure("Name","Selected cycles overlay + feature (Ch2)");
hold on; grid on;
for ii = 1:size(S2,1)
    plot(t_rel, S2(ii,:), "LineWidth", 0.6);
end
plot(t_rel, feat2, "k", "LineWidth", 3);
xlabel("time within cycle (s)"); ylabel("Ch2 (V, DC removed)");
title(sprintf("Ch2: %d selected cycles overlay + mean feature", size(S2,1)));

%% ---------------- Save selected 50 cycles + feature into ONE CSV (clear header) ----------------
outCSV_all = sprintf("Selected50_and_Feature_H%d.csv", H+1);

phase = linspace(0, 1, Lfeat).';

Tbig = table(phase, 'VariableNames', {'phase_0to1'});

% --- Ch1 cycles ---
for i = 1:size(S1,1)
    Tbig.(sprintf('CH1_cycle_%02d', i)) = S1(i,:).';
end

% --- Ch2 cycles ---
for i = 1:size(S2,1)
    Tbig.(sprintf('CH2_cycle_%02d', i)) = S2(i,:).';
end

% --- Features ---
Tbig.Feature_CH1 = feat1(:);
Tbig.Feature_CH2 = feat2(:);

% 直接写（第一行就是表头）
writetable(Tbig, outCSV_all);
fprintf("Saved ALL selected cycles + feature CSV: %s\n", outCSV_all);


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
        xr = x(:).';
        return;
    end
    u  = linspace(0,1,numel(x));
    uu = linspace(0,1,L);
    xr = interp1(u, x, uu, "linear", "extrap");
end

function score = ncc_score(x, tmpl)
    x = x(:);
    tmpl = tmpl(:);
    L = numel(tmpl);
    n = numel(x);
    if n < L
        error("Signal shorter than template.");
    end
    num = conv(x, flipud(tmpl), 'valid');  % length n-L+1
    winE = movsum(x.^2, [L-1 0]);
    winE = winE(L:end);
    den = sqrt(winE) * sqrt(sum(tmpl.^2));
    score = num ./ (den + 1e-12);
end

function y = shift_pad(x, lag)
    % lag>0: 向右移 lag 点；左侧用边界值填充（不环绕）
    x = x(:).';
    n = numel(x);
    y = zeros(1,n);

    if lag == 0
        y = x; return;
    elseif lag > 0
        y(1:lag) = x(1);
        y(lag+1:end) = x(1:end-lag);
    else
        lag = -lag;
        y(1:end-lag) = x(lag+1:end);
        y(end-lag+1:end) = x(end);
    end
end
function [selectedIdx, autoIdx] = select_cycles_click_ui(t, ch1, cycles, Npick)
    % 在原始波形上显示灰色背景 + 彩色周期，点击周期线条可选中/取消
    % 选中会变粗 + 变黑；上方显示已选数量
    % autoIdx：按 score 自动取 TopN

    numCycles = numel(cycles);
    if numCycles == 0
        selectedIdx = [];
        autoIdx = [];
        return;
    end

    sc = arrayfun(@(c)c.score, cycles);
    [~, ord] = sort(sc, "descend");
    autoIdx = ord(1:min(Npick, numCycles));

    fig = figure("Name","Click cycles to select (toggle). Then Done.", "NumberTitle","off");
    ax = axes(fig); hold(ax,"on"); grid(ax,"on");
    plot(ax, t, ch1, "Color",[0.7 0.7 0.7]); % 灰色背景
    xlabel(ax,"t (s)"); ylabel(ax,"Ch1 (V)");
    title(ax, sprintf("点击彩色周期选择/取消（目标 %d 个）。选中会变粗变黑。", Npick));

    C = lines(max(1,numCycles));
    hLine = gobjects(numCycles,1);

    for k = 1:numCycles
        s = cycles(k).s; e = cycles(k).e;
        col = C(mod(k-1,size(C,1))+1,:);
        hLine(k) = plot(ax, t(s:e), ch1(s:e), "LineWidth", 1.2, "Color", col);
        hLine(k).PickableParts = "all";
        hLine(k).HitTest = "on";
        hLine(k).UserData = k;
        hLine(k).ButtonDownFcn = @(src,evt)toggleSelect(src, fig, hLine);
    end

    % UI buttons
    btnW = 160; btnH = 28; gap = 10; left = 15; top = 15;
    hAuto = uicontrol(fig, "Style","pushbutton", "String","AutoTop50", ...
        "Position",[left, top, btnW, btnH], ...
        "Callback", @(src,evt)setAuto(fig, hLine, autoIdx));

    hClear = uicontrol(fig, "Style","pushbutton", "String","Clear", ...
        "Position",[left+btnW+gap, top, btnW, btnH], ...
        "Callback", @(src,evt)clearAll(fig, hLine));

    hDone = uicontrol(fig, "Style","pushbutton", "String","Done & Save Feature CSV", ...
        "Position",[left+2*(btnW+gap), top, 220, btnH], ...
        "Callback", @(src,evt)done(fig));

    hInfo = uicontrol(fig, "Style","text", "String","Selected: 0", ...
        "HorizontalAlignment","left", ...
        "Position",[left, top+btnH+6, 600, 20]);

    setappdata(fig, "selectedMask", false(numCycles,1));
    setappdata(fig, "hInfo", hInfo);

    uiwait(fig);

    if ~isvalid(fig)
        selectedIdx = [];
        return;
    end

    mask = getappdata(fig, "selectedMask");
    selectedIdx = find(mask);

    if isvalid(fig), close(fig); end

    % ---- nested callbacks ----
    function toggleSelect(src, f, hL)
        k = src.UserData;
        mask = getappdata(f, "selectedMask");
        mask(k) = ~mask(k);
        setappdata(f, "selectedMask", mask);
        refreshStyle(f, hL);
    end

    function refreshStyle(f, hL)
        mask = getappdata(f, "selectedMask");
        for kk = 1:numel(hL)
            if mask(kk)
                hL(kk).LineWidth = 2.8;
                hL(kk).Color = [0 0 0];
            else
                hL(kk).LineWidth = 1.2;
                % 恢复原色：从 lines 再算一次
                col = C(mod(kk-1,size(C,1))+1,:);
                hL(kk).Color = col;
            end
        end
        hI = getappdata(f, "hInfo");
        if isvalid(hI)
            hI.String = sprintf("Selected: %d (target %d)", nnz(mask), Npick);
        end
    end

    function setAuto(f, hL, idx)
        mask = false(numCycles,1);
        mask(idx) = true;
        setappdata(f, "selectedMask", mask);
        refreshStyle(f, hL);
    end

    function clearAll(f, hL)
        mask = false(numCycles,1);
        setappdata(f, "selectedMask", mask);
        refreshStyle(f, hL);
    end

    function done(f)
        uiresume(f);
    end
end

function [tmpl1_all, tmpl2_all, dur_all, fs] = pick_templates_ui(t, ch1s, ch2s, fs, Kmin, Kmax, cycleLen)
    tmpl1_all = [];
    tmpl2_all = [];
    dur_all   = [];
    k = 0;

    fig = figure("Name","Pick templates (Zoom then box-select)", "NumberTitle","off");
    ax = axes(fig);
    plot(ax, t, ch1s, "LineWidth", 1); grid(ax,"on");
    xlabel(ax,"t (s)"); ylabel(ax,"Ch1 (V)");
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
            try, zoom(fig, "off"); pan(fig, "off"); end
            set(hZoom, "Value", 0); set(hZoom, "String","Zoom OFF");
            set(hPan,  "Value", 0); set(hPan,  "String","Pan OFF");

            set(hInfo, "String", "状态：请用鼠标拖一个矩形框选一个周期（只取x范围）...");
            drawnow;

            r = drawrectangle(ax, "Color","r", "LineWidth", 1.5);
            wait(r);

            pos = r.Position;
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

            seg1r = resample_to_len(seg1, cycleLen);
            seg2r = resample_to_len(seg2, cycleLen);

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

    if isvalid(fig), close(fig); end

    function cbZoom(src, f)
        if src.Value == 1
            zoom(f, "on"); src.String = "Zoom ON";
        else
            zoom(f, "off"); src.String = "Zoom OFF";
        end
    end

    function cbPan(src, f, hZoomLocal)
        if src.Value == 1
            try, zoom(f, "off"); end
            if isvalid(hZoomLocal)
                hZoomLocal.Value = 0;
                hZoomLocal.String = "Zoom OFF";
            end
            pan(f, "on"); src.String = "Pan ON";
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
        try, xlim(axLocal, [tLocal(1), tLocal(end)]); end
    end
   


end
function idxWin = auto_spike_window(S, winLen)
    if isempty(S), idxWin = 1:size(S,2); return; end
    L = size(S,2);
    winLen = max(20, min(L, winLen));

    Sd = S - median(S,2);
    act = mean(abs(diff(Sd,1,2)), 1);
    act = [act, act(end)];

    [~,m] = max(act);
    a = max(1, m - floor(winLen/2));
    b = min(L, a + winLen - 1);
    a = max(1, b - winLen + 1);

    idxWin = false(1,L);
    idxWin(a:b) = true;
end

function [Sa, info] = align_by_spike(Sraw, idxWin, maxLag)
    [N,L] = size(Sraw);
    Sa = zeros(N,L);

    S = Sraw - median(Sraw,2);
    dx = diff(S,1,2);

    win = idxWin(1:end-1);
    w0 = find(win,1,'first'); if isempty(w0), win(:)=true; w0=1; end

    peakIdx = zeros(N,1);
    strength = zeros(N,1);
    sgn = zeros(N,1);

    for i = 1:N
        d = dx(i,win);
        [mx,imx] = max(d);
        [mn,imn] = min(d);

        if abs(mx) >= abs(mn)
            peakIdx(i) = w0 + imx - 1; strength(i) = abs(mx); sgn(i) = +1;
        else
            peakIdx(i) = w0 + imn - 1; strength(i) = abs(mn); sgn(i) = -1;
        end
    end

    target = round(median(peakIdx));

    for i = 1:N
        lag = peakIdx(i) - target;
        lag = max(-maxLag, min(maxLag, lag));
        Sa(i,:) = shift_pad(Sraw(i,:), -lag);
    end

    info.peakIdx = peakIdx;
    info.strength = strength;
    info.sign = sgn;
    info.target = target;
end

function s = mode_sign(x)
    if sum(x==+1) >= sum(x==-1), s = +1; else, s = -1; end
end

function y = weighted_mean(A, w)
    w = w(:);
    w = w / (sum(w) + 1e-12);
    y = (w.' * A);
end
