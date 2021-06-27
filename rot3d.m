function R = rot3d(t)
% 3d rotation matrix
    R = rotx(t(1)) * roty(t(2)) * rotz(t(3));
end

function Rx = rotx(t)
    Rx = [1 0 0; 
             0 cos(t) -sin(t); 
             0 sin(t) cos(t)] ;
end

function Ry = roty(t)
    Ry = [cos(t) 0 sin(t); 
              0 1 0 ; 
              -sin(t) 0  cos(t)] ;
end

function Rz = rotz(t)
    Rz = [cos(t) -sin(t) 0;
              sin(t) cos(t) 0;
              0 0 1];
end