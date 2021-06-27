function A = amatrix(bvec)
% diffusion tensor transformation matrix for given b vectors
A = [bvec(:,1) .* bvec(:,1),  2 * bvec(:,1) .* bvec(:,2),  2 * bvec(:,1) .* bvec(:,3), ...
        bvec(:,2) .* bvec(:,2),  2 * bvec(:,2) .* bvec(:,3),  bvec(:,3) .* bvec(:,3)];