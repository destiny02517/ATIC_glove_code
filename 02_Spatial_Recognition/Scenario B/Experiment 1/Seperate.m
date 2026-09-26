clc; clear; close all;

%% ============ 你需要改的参数 ============
startIdx = 1;      % 从 vofa+1 开始
endIdx   = 30;      % 到 vofa+5 结束（自己改大一点）

Ntrain = 70;
Nval   = 20;
Ntest  = 20;

rng(2025);         % 固定随机种子，保证可复现（不需要可删）

%% ============ 输入/输出根目录 ============
cycleRoot = fullfile(pwd, "Cycle");        % 你的周期切割输出根目录
outRoot   = fullfile(pwd, "Dataset");      % 新的数据集输出根目录

if ~exist(outRoot, "dir"), mkdir(outRoot); end

%% ============ 遍历变量（你要的）：vofa+1, vofa+2, vofa+3... ============
vofaBases = "vofa+" + string(startIdx:endIdx);   % <- 这个就是遍历用的变量

%% ============ 主循环：按“类”(每个 vofa) 拆分 train/val/test ============
for c = 1:numel(vofaBases)
    base = vofaBases(c);                             % 例如 "vofa+1"
    inDir = fullfile(cycleRoot, base);

    if ~exist(inDir, "dir")
        fprintf('[跳过] 找不到类文件夹: %s\n', inDir);
        continue;
    end

    % 找到该类下所有周期样本 CSV（例如 vofa+1_1.csv, vofa+1_2.csv...）
    files = dir(fullfile(inDir, base + "_*.csv"));
    if isempty(files)
        fprintf('[跳过] %s 下没有找到 %s_*.csv\n', inDir, base);
        continue;
    end

    % 样本总数
    nAll = numel(files);

    % 检查样本是否足够
    nNeed = Ntrain + Nval + Ntest;
    if nAll < nNeed
        fprintf('[警告] %s 样本数不足：有 %d，需要 %d。将按可用样本数缩放分配。\n', base, nAll, nNeed);

        % 按比例缩放（保持大致 60/20/20 的比例）
        rTrain = Ntrain / nNeed;
        rVal   = Nval   / nNeed;
        rTest  = Ntest  / nNeed;

        Ntrain_use = max(1, floor(nAll * rTrain));
        Nval_use   = max(1, floor(nAll * rVal));
        Ntest_use  = nAll - Ntrain_use - Nval_use;

        % 防止 test 变成 0
        if Ntest_use < 1
            Ntest_use = 1;
            if Ntrain_use > 1, Ntrain_use = Ntrain_use - 1;
            else, Nval_use = max(1, Nval_use - 1);
            end
        end
    else
        Ntrain_use = Ntrain;
        Nval_use   = Nval;
        Ntest_use  = Ntest;
    end

    % 打乱索引
    idx = randperm(nAll);

    idx_train = idx(1:Ntrain_use);
    idx_val   = idx(Ntrain_use+1 : Ntrain_use+Nval_use);
    idx_test  = idx(Ntrain_use+Nval_use+1 : Ntrain_use+Nval_use+Ntest_use);

    % 输出目录：Dataset/vofa+X/train|val|test
    outClassDir = fullfile(outRoot, base);
    outTrainDir = fullfile(outClassDir, "train");
    outValDir   = fullfile(outClassDir, "val");
    outTestDir  = fullfile(outClassDir, "test");

    if ~exist(outTrainDir, "dir"), mkdir(outTrainDir); end
    if ~exist(outValDir, "dir"),   mkdir(outValDir);   end
    if ~exist(outTestDir, "dir"),  mkdir(outTestDir);  end

    % 复制（或重写）CSV：每个CSV本身就是 351×40，不需要reshape
    copy_split(files, idx_train, inDir, outTrainDir);
    copy_split(files, idx_val,   inDir, outValDir);
    copy_split(files, idx_test,  inDir, outTestDir);

    fprintf('[完成] %s: train=%d, val=%d, test=%d -> %s\n', ...
        base, numel(idx_train), numel(idx_val), numel(idx_test), outClassDir);
end

%% ====== 本脚本用到的辅助函数（MATLAB R2016b+ 可直接放脚本末尾） ======
function copy_split(files, idx_list, inDir, outDir)
    for k = 1:numel(idx_list)
        f = files(idx_list(k));
        src = fullfile(inDir, f.name);
        dst = fullfile(outDir, f.name);
        copyfile(src, dst);   % 直接复制即可（内容仍是 40 路）
    end
end
