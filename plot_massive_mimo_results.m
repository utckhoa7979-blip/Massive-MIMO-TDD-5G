%% TRUC QUAN HOA KET QUA MASSIVE MIMO TDD
% Python thuc hien mo phong va tinh toan.
% MATLAB Online doc file Excel, ve va xuat cac do thi.
% Du lieu mo phong khong duoc tinh lai trong MATLAB.

clear;
clc;
close all;

%% 1. KHAI BAO FILE VA THU MUC

% Lay thu muc chua script MATLAB
scriptPath = mfilename("fullpath");
scriptDir = fileparts(scriptPath);

if isempty(scriptDir)
    scriptDir = pwd;
end

% File Excel nam cung thu muc voi script
fileName = fullfile(scriptDir, "MIMOTDD_MonteCarlo_Results.xlsx");
outputDir = fullfile(scriptDir, "Hinh_MATLAB");

if ~isfile(fileName)
    error("Khong tim thay file Excel: %s. Hay kiem tra MATLAB Drive.", fileName);
end

if ~isfolder(outputDir)
    mkdir(outputDir);
end

%% 2. DOC DU LIEU EXCEL

S = readtable(fileName, ...
    "Sheet", "Thong_ke_20_cau_hinh", ...
    "VariableNamingRule", "preserve");

D = readtable(fileName, ...
    "Sheet", "Du_lieu_chi_tiet", ...
    "VariableNamingRule", "preserve");

% Kiem tra cot cua bang thong ke
requiredSummary = [
    "Kich_ban"
    "M"
    "Phuong_phap"
    "SE_trung_binh"
    "SE_do_lech_chuan"
];

missingSummary = setdiff( ...
    requiredSummary, string(S.Properties.VariableNames));

if ~isempty(missingSummary)
    error("Thieu cot trong bang thong ke: %s", ...
        strjoin(missingSummary, ", "));
end

% Kiem tra cot cua bang du lieu chi tiet
requiredDetail = [
    "Scenario"
    "Configuration"
    "M"
    "MR_SINR"
    "ZF_SINR"
    "M-MMSE_SINR"
];

missingDetail = setdiff( ...
    requiredDetail, string(D.Properties.VariableNames));

if ~isempty(missingDetail)
    error("Thieu cot trong bang du lieu chi tiet: %s", ...
        strjoin(missingDetail, ", "));
end

%% 3. KHAI BAO KICH BAN VA PHUONG PHAP

scenarioNames = [
    "KB1_SingleCell", ...
    "KB2_MultiCell_f1", ...
    "KB3_MultiCell_f4"
];

scenarioLabels = [
    "KB1 - Don cell", ...
    "KB2 - Da cell, f = 1", ...
    "KB3 - Da cell, f = 4"
];

methods = ["MR", "ZF", "M-MMSE"];

sinrColumns = ["MR_SINR", "ZF_SINR", "M-MMSE_SINR"];

M_values = sort(unique(double(S.('M')))).';

fprintf("Doc file Excel thanh cong.\n");
fprintf("So dong thong ke: %d\n", height(S));
fprintf("So dong du lieu chi tiet: %d\n", height(D));

%% 4. THIET LAP DINH DANG DO THI

fontName = "Arial";
fontSize = 11;
titleSize = 13;

%% 5. DO THI SE CHO TUNG KICH BAN

% SE va do lech chuan duoc doc truc tiep tu ket qua Python.
% Thanh sai so the hien do lech chuan giua 20 cau hinh UE.

for iscen = 1:numel(scenarioNames)

    figureHandle = figure( ...
        "Color", "w", ...
        "Position", [100 100 900 560]);

    ax = axes(figureHandle);
    hold(ax, "on");

    scenarioMask = ...
        string(S.('Kich_ban')) == scenarioNames(iscen);

    Ts = S(scenarioMask, :);

    for im = 1:numel(methods)

        methodMask = ...
            string(Ts.('Phuong_phap')) == methods(im);

        Tm = Ts(methodMask, :);

        [x, order] = sort(double(Tm.('M')));

        y = double(Tm.('SE_trung_binh'));
        e = double(Tm.('SE_do_lech_chuan'));

        errorbar(ax, x, y(order), e(order), '-o', ...
            "LineWidth", 1.6, ...
            "MarkerSize", 5, ...
            "CapSize", 5, ...
            "DisplayName", char(methods(im)));
    end

    title(ax, ...
        "Hieu suat pho theo so anten - " + scenarioLabels(iscen), ...
        "FontName", fontName, ...
        "FontSize", titleSize, ...
        "FontWeight", "bold");

    xlabel(ax, "So anten tai tram goc, M", ...
        "FontName", fontName, "FontSize", fontSize);

    ylabel(ax, "Hieu suat pho trung binh (bit/s/Hz)", ...
        "FontName", fontName, "FontSize", fontSize);

    set(ax, "FontName", fontName, ...
        "FontSize", fontSize, "LineWidth", 0.9);

    xticks(ax, M_values);
    grid(ax, "on");
    box(ax, "on");

    legend(ax, "Location", "best", ...
        "FontName", fontName, "FontSize", 10);

    fileOut = fullfile(outputDir, ...
        "SE_" + scenarioNames(iscen) + ".png");

    exportgraphics(figureHandle, fileOut, "Resolution", 300);

end

%% 6. DO THI SINR CHO TUNG KICH BAN

% SINR trong Excel dang o thang tuyen tinh.
% Trong moi cau hinh UE, chuyen gia tri SINR da tong hop sang dB.
% Sau do tinh trung binh va do lech chuan tren 20 cau hinh.
%
% Luu y: day la thong ke cua SINR theo cau hinh,
% khong phai trung binh dB cua tung UE rieng le.

for iscen = 1:numel(scenarioNames)

    figureHandle = figure( ...
        "Color", "w", ...
        "Position", [100 100 900 560]);

    ax = axes(figureHandle);
    hold(ax, "on");

    scenarioMask = ...
        string(D.('Scenario')) == scenarioNames(iscen);

    Ds = D(scenarioMask, :);

    for im = 1:numel(methods)

        colName = char(sinrColumns(im));
        allSinrLinear = double(Ds.(colName));

        meanDb = nan(size(M_values));
        stdDb = nan(size(M_values));

        for imet = 1:numel(M_values)

            m = M_values(imet);

            maskM = double(Ds.('M')) == m;
            valsLinear = allSinrLinear(maskM);

            valsLinear = valsLinear( ...
                isfinite(valsLinear) & valsLinear > 0);

            valsDb = 10 .* log10(valsLinear);

            if ~isempty(valsDb)

                meanDb(imet) = mean(valsDb);

                if numel(valsDb) > 1
                    stdDb(imet) = std(valsDb, 0);
                else
                    stdDb(imet) = 0;
                end

            end
        end

        errorbar(ax, M_values, meanDb, stdDb, '-o', ...
            "LineWidth", 1.6, ...
            "MarkerSize", 5, ...
            "CapSize", 5, ...
            "DisplayName", char(methods(im)));

    end

    title(ax, ...
        "SINR theo so anten - " + scenarioLabels(iscen), ...
        "FontName", fontName, ...
        "FontSize", titleSize, ...
        "FontWeight", "bold");

    xlabel(ax, "So anten tai tram goc, M", ...
        "FontName", fontName, "FontSize", fontSize);

    ylabel(ax, "SINR trung binh theo cau hinh (dB)", ...
        "FontName", fontName, "FontSize", fontSize);

    set(ax, "FontName", fontName, ...
        "FontSize", fontSize, "LineWidth", 0.9);

    xticks(ax, M_values);
    grid(ax, "on");
    box(ax, "on");

    legend(ax, "Location", "best", ...
        "FontName", fontName, "FontSize", 10);

    fileOut = fullfile(outputDir, ...
        "SINR_dB_" + scenarioNames(iscen) + ".png");

    exportgraphics(figureHandle, fileOut, "Resolution", 300);

end

%% 7. SO SANH SE GIUA BA KICH BAN

figureHandle = figure( ...
    "Color", "w", ...
    "Position", [60 100 1500 540]);

tiledlayout(1, 3, ...
    "TileSpacing", "compact", ...
    "Padding", "compact");

for im = 1:numel(methods)

    ax = nexttile;
    hold(ax, "on");

    for iscen = 1:numel(scenarioNames)

        mask = ...
            string(S.('Kich_ban')) == scenarioNames(iscen) & ...
            string(S.('Phuong_phap')) == methods(im);

        Tm = S(mask, :);

        [x, order] = sort(double(Tm.('M')));
        y = double(Tm.('SE_trung_binh'));

        plot(ax, x, y(order), '-o', ...
            "LineWidth", 1.6, ...
            "MarkerSize", 4, ...
            "DisplayName", char(scenarioLabels(iscen)));

    end

    title(ax, "Phuong phap " + methods(im), ...
        "FontName", fontName, ...
        "FontSize", 12, ...
        "FontWeight", "bold");

    xlabel(ax, "So anten tai tram goc, M", ...
        "FontName", fontName, "FontSize", fontSize);

    if im == 1
        ylabel(ax, "Hieu suat pho trung binh (bit/s/Hz)", ...
            "FontName", fontName, "FontSize", fontSize);
    end

    set(ax, "FontName", fontName, ...
        "FontSize", 10, "LineWidth", 0.9);

    xticks(ax, M_values);
    grid(ax, "on");
    box(ax, "on");

    legend(ax, "Location", "best", ...
        "FontName", fontName, "FontSize", 9);

end

sgtitle("So sanh hieu suat pho giua ba kich ban Massive MIMO TDD", ...
    "FontName", fontName, ...
    "FontSize", 14, ...
    "FontWeight", "bold");

exportgraphics(figureHandle, ...
    fullfile(outputDir, "So_sanh_3_kich_ban_SE.png"), ...
    "Resolution", 300);

%% 8. SO SANH SINR GIUA BA KICH BAN

figureHandle = figure( ...
    "Color", "w", ...
    "Position", [60 100 1500 540]);

tiledlayout(1, 3, ...
    "TileSpacing", "compact", ...
    "Padding", "compact");

for im = 1:numel(methods)

    ax = nexttile;
    hold(ax, "on");

    for iscen = 1:numel(scenarioNames)

        maskScenario = ...
            string(D.('Scenario')) == scenarioNames(iscen);

        Ds = D(maskScenario, :);

        colName = char(sinrColumns(im));
        valsLinearAll = double(Ds.(colName));

        meanDb = nan(size(M_values));

        for imet = 1:numel(M_values)

            m = M_values(imet);

            maskM = double(Ds.('M')) == m;
            valsLinear = valsLinearAll(maskM);

            valsLinear = valsLinear( ...
                isfinite(valsLinear) & valsLinear > 0);

            valsDb = 10 .* log10(valsLinear);

            if ~isempty(valsDb)
                meanDb(imet) = mean(valsDb);
            end

        end

        plot(ax, M_values, meanDb, '-o', ...
            "LineWidth", 1.6, ...
            "MarkerSize", 4, ...
            "DisplayName", char(scenarioLabels(iscen)));

    end

    title(ax, "Phuong phap " + methods(im), ...
        "FontName", fontName, ...
        "FontSize", 12, ...
        "FontWeight", "bold");

    xlabel(ax, "So anten tai tram goc, M", ...
        "FontName", fontName, "FontSize", fontSize);

    if im == 1
        ylabel(ax, "SINR trung binh theo cau hinh (dB)", ...
            "FontName", fontName, "FontSize", fontSize);
    end

    set(ax, "FontName", fontName, ...
        "FontSize", 10, "LineWidth", 0.9);

    xticks(ax, M_values);
    grid(ax, "on");
    box(ax, "on");

    legend(ax, "Location", "best", ...
        "FontName", fontName, "FontSize", 9);

end

sgtitle("So sanh SINR giua ba kich ban Massive MIMO TDD", ...
    "FontName", fontName, ...
    "FontSize", 14, ...
    "FontWeight", "bold");

exportgraphics(figureHandle, ...
    fullfile(outputDir, "So_sanh_3_kich_ban_SINR_dB.png"), ...
    "Resolution", 300);

%% 9. THONG BAO KET QUA

fprintf("\nDa tao xong 8 do thi.\n");
fprintf("Thu muc luu hinh: %s\n", outputDir);