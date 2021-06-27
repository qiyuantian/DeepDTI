function dtimetrics = decompose_tensor(tensor, mask)
% decompose diffusion tensor to derive dti metrics 
mask = mask > 0.1;
sz = size(mask);

v1 = zeros([sz, 3]);
v2 = zeros([sz, 3]);
v3 = zeros([sz, 3]);
l1 = zeros(sz);
l2 = zeros(sz);
l3 = zeros(sz);
md = zeros(sz);
rd = zeros(sz);
fa = zeros(sz);

for ii = 1 : size(mask, 1)
    for jj = 1 : size(mask, 2)
        for kk = 1 : size(mask, 3)
            if mask(ii, jj, kk)
                tensor_vox = squeeze(tensor(ii, jj, kk, :)); 
                
                tensor_mtx = zeros(3, 3);
                tensor_mtx(1, 1) = tensor_vox(1);
                tensor_mtx(1, 2) = tensor_vox(2); tensor_mtx(2, 1) = tensor_vox(2);
                tensor_mtx(1, 3) = tensor_vox(3); tensor_mtx(3, 1) = tensor_vox(3);
                tensor_mtx(2, 2) = tensor_vox(4);
                tensor_mtx(2, 3) = tensor_vox(5); tensor_mtx(3, 2) = tensor_vox(5);
                tensor_mtx(3, 3) = tensor_vox(6);
                
                [V, D] = eig(tensor_mtx); % v3, v2, v1, primary orientations come last
                D = sum(D, 2);
                MD = mean(D);
                FA = sqrt(sum((D - MD).^2)) ./ sqrt(sum(D.^2)) * sqrt(1.5);

                v1(ii, jj, kk, :) = V(:, 3);
                v2(ii, jj, kk, :) = V(:, 2);
                v3(ii, jj, kk, :) = V(:, 1);
                l1(ii, jj, kk) = D(3);
                l2(ii, jj, kk) = D(2);
                l3(ii, jj, kk) = D(1);
                fa(ii, jj, kk) = FA;
                md(ii, jj, kk) = MD;
                rd(ii, jj, kk) = mean(D(1:2));
            end
        end
    end
end

dtimetrics.v1 = v1;
dtimetrics.v2 = v2;
dtimetrics.v3 = v3;
dtimetrics.l1 = l1;
dtimetrics.l2 = l2;
dtimetrics.l3 = l3;
dtimetrics.fa = fa;
dtimetrics.md = md;
dtimetrics.rd = rd;


