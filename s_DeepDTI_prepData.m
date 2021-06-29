%% introduction

% s_DeepDTI_prepData.m
% 
%   A script for preparing input and output data for convolutional neural
%   network in DeepDTI.
%
%   Source code:
%       https://github.com/qiyuantian/DeepDTI/blob/main/s_DeepDTI_prepData.m
%
%   HTML file can be automatically generaged using command:
%       publish('s_DeepDTI_prepData.m', 'html');
%
%   Reference:
%       [1] Tian Q, Bilgic B, Fan Q, Liao C, Ngamsombat C, Hu Y, Witzel T,
%       Setsompop K, Polimeni JR, Huang SY. DeepDTI: High-fidelity
%       six-direction diffusion tensor imaging using deep learning.
%       NeuroImage. 2020;219:117017. 
%
%       [2] Tian Q, Li Z, Fan Q, Ngamsombat C, Hu Y, Liao C, Wang F,
%       Setsompop K, Polimeni JR, Bilgic B, Huang SY. SRDTI: Deep
%       learning-based super-resolution for diffusion tensor MRI. arXiv
%       preprint. 2021; arXiv:2102.09069.
%
% (c) Qiyuan Tian, Harvard, 2021

%% load data

clear, clc, close all

tmp = load('data.mat');
data = double(tmp.data); % 18 b = 0 volumes and 90 dwi volumes, only 70 axial slices
% image data were corrected for intensity bias already, can use dwibiascorrect from MRtrix3
% we did not evaluate the effects of bias on the performance of CNN
bvals = tmp.bvals; % b = 1000 s/mm^2
bvecs = tmp.bvecs;
mask = tmp.mask;

b0s = data(:, :, :, bvals < 100); % all 18 b = 0 images
meanb0 = mean(b0s, 4); % mean b = 0 images
dwis = data(:, :, :, bvals > 100); % all 90 dwis
bvals_dwi = bvals(bvals > 100); % bvals for dwis
bvecs_dwi = bvecs(bvals > 100, :); % bvecs for dwis

dirs = bvecs(bvals > 100, :); % 90 diffusion-encoding directions
dirs_vis = dirs .* sign(dirs(:, 3)); % directions flipped to z > 0 for visualization
sz_data = size(data);

figure; % display image data
subplot(1, 2, 1)
imshow(rot90(data(:, :, 35, 1)), [0, 10000])
title('b = 0 image');
subplot(1, 2, 2)
imshow(rot90(data(:, :, 35, 2)), [0, 4000])
title('diffusion weighted image');

figure; % display diffusion encoding directions, all flipped to z > 0 
plot3(dirs_vis(:, 1), dirs_vis(:, 2), dirs_vis(:, 3), '.');
grid on, axis equal
zlim([-1, 1])
title('90 diffusion encoding directions');

%% select optimized encoding directions

% 6 optimized directions from the DSM scheme that minimizes the condition
% number of the diffusion tensor transformation matrix 
% from S Skare et al., J Magn Reson. 2000;147(2):340-52
dsm6 = [0.91, 0.416, 0; ...
               0, 0.91, 0.416; ...
               0.416, 0, 0.91; ...
               0.91, -0.416, 0; ...
               0, 0.91, -0.416; ...
              -0.416, 0, 0.91];

dsm6_norm = dsm6 ./ sqrt(dsm6(:, 1) .^ 2 + dsm6(:, 2) .^ 2 + dsm6(:, 3) .^ 2); % normalize vectors

% randomly rotate the DSM6 dirs, select their nearest 6 dirs from acquired
% dirs, keep those with low condition number and angel difference
rotang_all = [];
angerr_all  = [];
condnum_all = [];
ind_all = [];
for ii = 1 : 100000 % number of iterations can be increased
    
    rotangs = rand(1, 3) * 2 * pi; % random angles to rotate around x, y, z axis
    R = rot3d(rotangs); % rotation matrix
    dsm6_rot = (R * dsm6_norm')'; % roated directions
   
    % find 6 nearest dirs in acquired dirs
    angerrors = acosd(abs(dsm6_rot * dirs')); % angles btw rotated DSM6 dirs and acquired dirs
    [minerrors, ind] = min(angerrors, [], 2); % 6 dirs with min angles compared to rotated DSM6 dirs

    meanangerr = mean(minerrors); % mean angle errors of selected dirs
    condnum = cond(amatrix(dirs(ind, :))); % cond number of tensor tx matrix of selected dirs
    
    if meanangerr < 5 && condnum < 1.6 % only use dirs with low angle error and cond number
        if isempty(ind_all) || ~any(sum(ind_all == sort(ind'), 2) == 6) % make sure no repetition
            
            % record params for satisfied sets
            angerr_all = cat(1, angerr_all, meanangerr);
            condnum_all = cat(1, condnum_all, condnum);
            ind_all = cat(1, ind_all, sort(ind'));
            rotang_all = cat(1, rotang_all, rotangs);
        end
    end
end

% here only select 5 sets of directions with lowest condition number
% better to make sure all dwis can be equally slected and used for training 
[~, ind_sort] = sort(condnum_all);
ind_use = ind_all(ind_sort(1 : 5), :);
condnum_use = condnum_all(ind_sort(1 : 5));
angerr_use = angerr_all(ind_sort(1 : 5));
rotang_use = rotang_all(ind_sort(1 : 5), :);

figure; % display two selected sets of 6 optimal directions
for ii = 1 : 2
    subplot(1, 2, ii)
    plot3(dirs_vis(:, 1), dirs_vis(:, 2), dirs_vis(:, 3), '.'); % all dirs
    hold on
    
    visdirs_use = dirs_vis(ind_use(ii, :), :);
    plot3(visdirs_use(:, 1), visdirs_use(:, 2), visdirs_use(:, 3), 'o'); % selected dirs
    
    R = rot3d(rotang_use(ii, :));
    dsm6_rot = (R * dsm6_norm')';
    dsm6_rot_vis = dsm6_rot .* sign(dsm6_rot(:, 3));
    plot3(dsm6_rot_vis(:, 1), dsm6_rot_vis(:, 2), dsm6_rot_vis(:, 3), 'x'); % rotated DSM6 dirs
    
    grid on, axis equal
    xlim([-1, 1])
    ylim([-1, 1])
    zlim([-1, 1])
    title(['cond num=' num2str(condnum_use(ii))]);
end

%% generate input data of CNN
input_all = {};
tensor_all = {};
bval_synth = 1000; % b value for synthesized dwis

for ii = 1 : 5
    b0 = b0s(:, :, :, ii); % a single b=0 image for each set of selected 6 dwis
    dwis6 = dwis(:, :, :, ind_use(ii, :));
    bvals6 = bvals_dwi(ind_use(ii, :));
    bvecs6 = bvecs_dwi(ind_use(ii, :), :);
    
    % compute apparent diffusion coefficients
    adcs6 = log(dwis6 ./ b0); % s = s0 * exp(-b * adc) 
    for jj = 1 : length(bvals6)
        adcs6(:, :, :, jj) = adcs6(:, :, :, jj) / (-bvals6(jj)); % in case bvalues might be different for acquired dwis
    end
    
    % compute tensors
    adcs6_vec = reshape(adcs6, sz_data(1)*sz_data(2)*sz_data(3), size(adcs6, 4)); % transform volume data to vectors
    A = amatrix(bvecs6); % tensor transformation matrix
    tensor_vec = A \ adcs6_vec'; %i.e., inv(A) * adcs_vec'; solve Ax = b
    tensor = reshape(tensor_vec', [sz_data(1:3), 6]); % transform tensors in vector form to volume
    tensor(isnan(tensor)) = 0;
    tensor(isinf(tensor)) = 0;
    tensor_all{ii} = tensor;

    % synthesize dwis along DSM6 dirs
    dwis6norm_vec_synth = exp(-bval_synth .* amatrix(dsm6_norm) * tensor_vec); % normalized dwi
    dwis6_synth = b0 .* reshape(dwis6norm_vec_synth', [sz_data(1:3), size(dwis6norm_vec_synth, 1)]);    
    dwis6_synth(isnan(dwis6_synth)) = 0;
    dwis6_synth(isinf(dwis6_synth)) = 0;
    
    diff_input = cat(4, b0, dwis6_synth);
    input_all{ii} = diff_input;
end

figure; % display synthesized dwis
for ii = 1 : 2
    diff_input = input_all{ii};
    subplot(1, 2, ii)
    imshow(rot90(diff_input(:, :, 35, 2)), [0, 4000])
    title('synthsized dwi');
end

figure; % display synthesized dwis
for ii = 1 : 2
    tensor = tensor_all{ii};
    subplot(1, 2, ii)
    
    dtimetrics = decompose_tensor(tensor, mask);
    fa = dtimetrics.fa;
    imshow(rot90(fa(:, :, 35)), [0, 1])
    title('fractional anisotropy');
end

%% generate ground-truth data of CNN

% compute apparent diffusion coefficients
adcs = log(dwis ./ meanb0); % s = b0 * exp(-b * adc)
for ii = 1 : size(adcs, 4)
    adcs(:, :, :, ii) = adcs(:, :, :, ii) / (-bvals_dwi(ii)); % in case bvalues might be different for acquired dwis
end

adcs_vec = reshape(adcs, sz_data(1)*sz_data(2)*sz_data(3), size(adcs, 4)); % tx volume data to vectors
tensor_gt_vec = amatrix(bvecs_dwi) \ adcs_vec'; % solve tensors
tensor_gt = reshape(tensor_gt_vec', [sz_data(1:3), 6]); % tx vectors to volume
tensor_gt(isnan(tensor_gt)) = 0;
tensor_gt(isinf(tensor_gt)) = 0;

% synthesize dwis along DSM6 dirs
dwis6norm_vec_gt = exp(-bval_synth .* amatrix(dsm6_norm) * tensor_gt_vec);
dwis6_gt = b0 .* reshape(dwis6norm_vec_gt', [sz_data(1:3), size(dwis6norm_vec_gt, 1)]);    
dwis6_gt(isnan(dwis6_gt)) = 0;
dwis6_gt(isinf(dwis6_gt)) = 0;

diff_gt = cat(4, meanb0, dwis6_gt);

figure; % display ground-truth dwi and fa
subplot(1, 2, 1)
imshow(rot90(diff_gt(:, :, 35, 2)), [0, 4000])
title('ground-truth dwi');
    
dtimetrics = decompose_tensor(tensor_gt, mask);
fa = dtimetrics.fa;
subplot(1, 2, 2)
imshow(rot90(fa(:, :, 35)), [0, 1])
title('ground-truth fa');

figure; % display residuals btx input dwis and ground truth
for ii = 1 : 2
    diff_input = input_all{ii};
    subplot(1, 2, ii)
    imagesc(rot90(diff_gt(:, :, 35, 2)) - rot90(diff_input(:, :, 35, 2)), [-1000, 1000])
    axis image
    title('residual image');
    colormap(bgr_colormap);
    colorbar;
end

%% save data

% data is saved in unit16 to save sapce to be shared on Github
% actual implemenation should use floating point to maintain precision
% these input and output data can be used to train any CNN for denoising
diff_input1 = uint16(input_all{1});
diff_input2 = uint16(input_all{2});
diff_input3 = uint16(input_all{3});
diff_input4 = uint16(input_all{4});
diff_input5 = uint16(input_all{5});
diff_gt = uint16(diff_gt);

save('cnn_inout.mat', 'diff_input1', 'diff_input2', 'diff_input3', ...
         'diff_input4', 'diff_input5', 'diff_gt', 'mask');
