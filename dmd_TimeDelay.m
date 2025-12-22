%% ================ TIMEDELAY DMD TRAINING ================
clc; clear all; close all;
tic;

%% ========================= CONFIGURAZIONE =========================
% Parametri Modificabili
H = 3;                  % Time Delay Horizon (t, t-1, t-2)
target_dim = [30 30];   % RIDUZIONE NECESSARIA! (es. 30x30 o 40x40)
% Nota: Se usi 100x100, lo stato aumentato è 30.000. La matrice A sarebbe 30.000x30.000 (Troppo grande!)
% Con 30x30 -> Stato 900 -> Stato Aumentato 2700. Matrice A ~ 50 MB. Perfetto.

path = './Simulazioni/';

%% ======================== DATA LOAD ========================
subdirs = dir(fullfile(path)); 
subdirs = subdirs([subdirs.isdir]);
subdirs = subdirs(~ismember({subdirs.name}, {'.', '..'}));

fprintf('🗂️ Trovate %d cartelle di simulazione.\n', length(subdirs));

% Riordino naturale (sim1, sim2...)
names = {subdirs.name};
numVals = zeros(size(names));
for k = 1:numel(names)
    num = sscanf(names{k}, 'sim%d');
    if isempty(num), num = inf; end
    numVals(k) = num;
end
[~, idx] = sort(numVals);
subdirs = subdirs(idx);

% Separazione Train/Test (Test = simFinale)
% Cerchiamo la cartella "SimFinale"
final_idx = find(strcmpi({subdirs.name}, 'SimFinale'));
if ~isempty(final_idx)
    % Divisione: simFinale nel test set, le altre nel training set
    % Se esiste, la mettiamo da parte come TEST SET
    % setdiff(1:N, k) prende tutti gli indici tranne k (quello di SimFinale)
    train_dirs = subdirs(setdiff(1:numel(subdirs), final_idx));
    test_dirs  = subdirs(final_idx); % mettiamo SimFinale nel test set
    fprintf('Divisione Dataset: %d Train, 1 Test (SimFinale)\n', numel(train_dirs));
else
    % Se non esiste, usiamo tutto per il training
    train_dirs = subdirs;
    test_dirs  = [];
    fprintf('AVVISO: "SimFinale" non trovata. Divisione Dataset: %d Train, 0 Test\n', numel(train_dirs));
end

fprintf('\n📜 Divisione simulazioni:\n');
fprintf(' --> %d nel training set\n', numel(train_dirs));
fprintf(' --> %d nel test set (simFinale)\n\n', numel(test_dirs));
disp({subdirs.name});

fprintf('🏋️‍♂️ Elaborazione TRAIN SET con Time Delay H=%d...\n', H);
fprintf('📏 Risoluzione target: %dx%d\n', target_dim(1), target_dim(2));

X_aug = []; % Stato aumentato corrente [x(k-2); x(k-1); x(k)]
Y_aug = []; % Stato aumentato futuro   [x(k-1); x(k); x(k+1)]
U_aug = []; % Controllo u(k)

pixels = prod(target_dim);
valid_train = 0;

for j = 1:length(train_dirs)
    subdir_name = train_dirs(j).name;
    subdir_path = fullfile(path, subdir_name);
    
    % 1. Carica Immagini
    imgfile = dir(fullfile(subdir_path, '*.png'));
    if isempty(imgfile), continue; end
    
    % Ordina frame per numero
    fnames = {imgfile.name};
    frame_ids = zeros(1, numel(fnames));
    for kk = 1:numel(fnames)
        tok = regexp(fnames{kk}, '(\d+)', 'match');
        if ~isempty(tok), frame_ids(kk) = str2double(tok{end}); end
    end
    [frame_ids_sorted, sort_idx] = sort(frame_ids);
    fnames = fnames(sort_idx);
    
    % 2. Carica Controllo U
    simdata_path = fullfile(subdir_path, 'sim_data.mat');
    if ~isfile(simdata_path), continue; end
    U_data = load(simdata_path);
    ups = U_data.u; % 2 x N
    
    % 3. Leggi e Processa Immagini
    num_imgs = length(fnames);
    % Ci servono almeno H+1 immagini per fare uno step
    if num_imgs < H+1, continue; end
    
    % Matrice temporanea immagini della singola simulazione
    % Limitiamo al numero di input disponibili
    T_eff = min(num_imgs, size(ups, 2)); 
    temp_imgs = zeros(pixels, T_eff);
    
    for i = 1:T_eff
        img_path = fullfile(subdir_path, fnames{i});
        img = imread(img_path);
        img = imresize(img, target_dim);
        if size(img,3)==3, img=rgb2gray(img); end
        temp_imgs(:, i) = double(img(:))/255.0;
    end
    
    % 4. Costruisci lo Stack Temporale (Sliding Window)
    % Stato k va da H a T_eff-1
    % Esempio H=3.
    % k=3: Stato=[Img1; Img2; Img3], Futuro=[Img2; Img3; Img4], u=u(3)
    
    for k = H : (T_eff - 1)
        % Costruisco colonne impilate
        stack_k = [];
        stack_kp1 = [];
        
        for delay = (H-1):-1:0
            stack_k   = [stack_k;   temp_imgs(:, k - delay)];
            stack_kp1 = [stack_kp1; temp_imgs(:, k+1 - delay)];
        end
        
        u_k = ups(:, k); % Controllo al tempo k
        
        X_aug = [X_aug, stack_k];
        Y_aug = [Y_aug, stack_kp1];
        U_aug = [U_aug, u_k];
    end
    
    valid_train = valid_train + 1;
    if mod(j, 10)==0, fprintf('..proc %d folders\n', j); end
end

fprintf('✅ Dataset Costruito. Campioni totali: %d\n', size(X_aug, 2));
fprintf('Dimensione Stato Aumentato: %d\n', size(X_aug, 1));

%% ========================= CALCOLO DMD (FULL) =========================
% Risolviamo Y = A*X + B*U -> Y = [A B] * [X; U]
disp('🧮 Calcolo Pseudo-Inversa (può richiedere tempo)...');

% Metodo Robusto (Backslash operator)
Omega = [X_aug; U_aug];
% AB = Y_aug / Omega; % Y * pinv(Omega)
% Per risparmiare memoria usiamo la trasposta: (Omega' \ Y')'
AB_trans = Omega' \ Y_aug';
AB = AB_trans';

nu = 2;
Atilde = AB(:, 1:end-nu);
Btilde = AB(:, end-nu+1:end);

%% ======================== SALVATAGGIO ========================
outfile = 'dynamics_timedelay.mat';
save(outfile, 'Atilde', 'Btilde', 'H', 'target_dim');
fprintf('💾 Matrici salvate in %s\n', outfile);

%% ====================== VERIFICA ERRORI ======================
disp('📈 Verifica Errore Ricostruzione (Training Set)...');
Y_pred = Atilde * X_aug + Btilde * U_aug;
Error = mean(mean(abs(Y_aug - Y_pred)));
fprintf('Mean Absolute Reconstruction Error: %.6f\n', Error);

%% ====================== VERIFICA SU TEST SET (SimFinale) ======================
if ~isempty(test_dirs)
    fprintf('\n🧪 Verifica Modello su TEST SET (%s)...\n', test_dirs(1).name);
    
    % --- 1. Caricamento e Preparazione Dati Test ---
    subdir_path = fullfile(path, test_dirs(1).name);
    simdata = load(fullfile(subdir_path, 'sim_data.mat'));
    U_test_raw = simdata.u;
    
    imgfiles = dir(fullfile(subdir_path, '*.png'));
    fnames = {imgfiles.name};
    fids = zeros(1,numel(fnames));
    for k=1:numel(fnames)
        tok = regexp(fnames{k}, '(\d+)', 'match');
        if ~isempty(tok), fids(k)=str2double(tok{end}); end
    end
    [~, idx] = sort(fids);
    fnames = fnames(idx);
    
    N_test = min(length(fnames), size(U_test_raw, 2));
    pixels = prod(target_dim);
    temp_imgs = zeros(pixels, N_test);
    
    fprintf('Caricamento %d immagini di test...\n', N_test);
    for i=1:N_test
        im = imread(fullfile(subdir_path, fnames{i}));
        im = imresize(im, target_dim);
        if size(im,3)==3, im=rgb2gray(im); end
        temp_imgs(:,i) = double(im(:))/255.0;
    end
    
    % --- 2. Costruzione Stack Time-Delay (H) ---
    X_test_aug = [];
    Y_test_aug = []; 
    U_test_aug = [];
    
    for k = H : (N_test - 1)
        stack_k = [];
        stack_kp1 = [];
        for delay = (H-1):-1:0
            stack_k   = [stack_k;   temp_imgs(:, k - delay)];
            stack_kp1 = [stack_kp1; temp_imgs(:, k+1 - delay)];
        end
        X_test_aug = [X_test_aug, stack_k];
        Y_test_aug = [Y_test_aug, stack_kp1];
        U_test_aug = [U_test_aug, U_test_raw(:, k)];
    end
    
    % --- 3. Predizione col Modello DMD ---
    fprintf('Predizione su %d passi...\n', size(X_test_aug, 2));
    Y_test_pred = Atilde * X_test_aug + Btilde * U_test_aug;
    
    % --- 4. Calcolo e Plot Errore ---
    error_time = mean(abs(Y_test_aug - Y_test_pred), 1);
    fprintf('Error Medio su Test Set: %.6f\n', mean(error_time));
    
    figure('Name', 'Errore Test Set');
    plot(error_time, 'LineWidth', 1.5, 'Color', '#D95319');
    title(['Errore Ricostruzione Time-Delay su ', test_dirs(1).name]);
    xlabel('Time Step'); ylabel('MAE (Pixel)'); grid on;
    saveas(gcf, 'Error_TestSet_TimeDelay.png');
    
    %% ================= GENERAZIONE GIF PREDIZIONE =================
    gif_filename = 'dmd_timedelay_prediction.gif';
    fprintf('🎥 Generazione GIF predizione (%s)...\n', gif_filename);
    
    % Usiamo una figura invisibile per velocità
    h_gif = figure('Visible', 'off'); 
    axis tight manual; axis off;
    
    T_pred = size(Y_test_pred, 2);
    
    for k = 1:T_pred
        % --- IL TRUCCO: Estrazione dell'immagine futura dallo stack ---
        % Il vettore Y_test_pred(:,k) è alto H*pixels.
        % L'immagine al tempo t+1 sono gli ULTIMI 'pixels' elementi.
        aug_vec = Y_test_pred(:, k);
        img_vec_future = aug_vec(end - pixels + 1 : end);
        
        % Reshape per visualizzare
        img_pred = reshape(img_vec_future, target_dim(1), target_dim(2));
        
        imshow(img_pred, [0 1]); % [0 1] fissa la scala di grigi
        title(sprintf('Predizione TD H=%d - Frame %d', H, k));
        drawnow;
        
        % Salvataggio frame per GIF
        frame = getframe(h_gif);
        im = frame2im(frame);
        [imind, cm] = rgb2ind(im, 256);
        if k == 1
            imwrite(imind, cm, gif_filename, 'gif', 'LoopCount', inf, 'DelayTime', 0.04);
        else
            imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.04);
        end
    end
    close(h_gif);
    fprintf('✅ GIF salvata con successo!\n');
else
    disp('⚠️ Nessuna cartella SimFinale trovata per il test.');
end

elapsed_time = toc;
fprintf('⏳ Durata totale: %.2f sec\n', elapsed_time);