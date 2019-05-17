function [PR] = process(directory, filesNumber, processeDataset, min_consecutive_loops, nAgents, gt_neigh, compensate, stp)

    % Establishing default parameters
    if nargin < 6
        gt_neigh = 20;
        compensate = false;
    elseif nargin == 6
        compensate = false;
    end
    robots = 0;
    for fileNumber = 0:stp:filesNumber
        
        if fileNumber == 0
            robots = 1;
        else
            robots = fileNumber;
        end
        % Getting dataset information and loading files
        loops_filename = strcat(directory, int2str(robots), '.txt');
        loops_file = load(loops_filename);

        % Reading info file    
%         info_filename = strcat(directory, 'info.json');
%         fid = fopen(info_filename);
%         raw = fread(fid,inf);
%         str = char(raw');
%         fclose(fid);
%         json_info = jsondecode(str);

        % Ground truth  
        if strcmp(processeDataset, 'Lip6In')
            gt_filename = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets/Lip6_indoor/groundtruth.mat';
%             coords_filename = '';
        elseif strcmp(processeDataset, 'Lip6Out')
            gt_filename = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets/Lip6_outdoor/groundtruth.mat';
%             coords_filename = '';
        elseif strcmp(processeDataset, 'CityCentre')
            gt_filename = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets/CityCentre/groundtruth.mat';
%             coords_filename = '/home/mamlpm/Documentos/TrabajoFinMast/ImageCollectionCoordinates.mat';
        elseif strcmp(processeDataset, 'KITTI00')
            gt_filename = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets/KITTI/00/groundtruth.mat';
%             coords_filename = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets/KITTI/00/imageCoords.mat';
        else
            gt_filename = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets/KITTI/05/groundtruth.mat';
%             coords_filename = '/home/mamlpm/Documentos/TrabajoFinMaster/datasets/KITTI/05/imageCoords05.mat';
        end
        
        gt_file = load(gt_filename);

        % Coordinate file    
%         coords_filename = json_info.coords_file;
        %coords_file = load(coords_filename);

        % Reading parameters
        cons_loops = min_consecutive_loops;
        %inliers = json_info.min_inliers;

        % Obtaining P/R Curve varying the number of inliers 
        P = [1.0];
        R = [0.0];
        P_max = 0.0;
        R_max = 0.0;
        I_max = 0;
        for i=1:500
            % Processing the resulting file to transform the format
            loops_trans_file = detect_loops(loops_file, cons_loops, i, robots);
            [Pr, Re] = compute_PR(loops_trans_file, gt_file, gt_neigh, compensate, false);
            P = [P, Pr];
            R = [R, Re];            

            if Pr > P_max || (Pr == P_max && Re > R_max)
                P_max = Pr;
                R_max = Re;
                I_max = i;
            end
        end

        % Ordering the obtained results    
        [R, I] = sort(R);
        P_a = P;    
        for i=1:numel(I)
            P_a(i) = P(I(i));
        end
        P = P_a;

        % Filtering resulting points
        P_a = [P(1)];
        R_a = [R(1)];
        for i=2:numel(P)
            if P(i) <= P_a(end)
                P_a = [P_a, P(i)];
                R_a = [R_a, R(i)];
            end
        end
        P = P_a;
        R = R_a;

        % Returning the information for P/R
        PR(robots).P = P;
        PR(robots).R = R;
        PR(robots).P_max = P_max;
        PR(robots).R_max = R_max;
        PR(robots).I_max = I_max;

        % Computing response and index size vectors
%         curr_loops_size = size(loops_file);
%         nimages = curr_loops_size(1);
%         imgvssize.img = [];
%         imgvssize.size = [];
%         imgvstime.img = [];
%         imgvstime.time = [];
%         for i=1:nimages
            % Images vs Index Size
%             imgvssize.img(i) = i;
%             imgvssize.size(i) = loops_file(i, 6);

            % Images vs Time
%             imgvstime.img(i) = i;
%             imgvstime.time(i) = loops_file(i, 7);
        end
end