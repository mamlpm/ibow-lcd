%
% This file is part of htmap.
%
% Copyright (C) 2018 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
%
% htmap is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% htmap is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with htmap. If not, see <http://www.gnu.org/licenses/>.

function plot_topmap(tmap_file, coords_file, datasetname)
    tmap = load(tmap_file);
    coords = load(coords_file);
    if datasetname == 'CC'
        % For CC and NC
        coords = coords.GPS(1:2:end, :);
    else
        % For KITTIs
        c = coords.p;
        % For KITTI 06
        %c(:, 1) = coords.p(:, 2);
        %c(:, 2) = -coords.p(:, 1);
        % For KITTIs
        coords = c;       
        % For St. Lucia
        %coords = coords.fGPS(1:end, :);
    end
    nimages = length(tmap);
    
    total_nodes = max(tmap(:,4)) + 1;
    
    %cmap = prism(total_nodes);
    cmap = rand(total_nodes, 3);
    
    figure()
    hold on;
    for i=2:nimages        
        if tmap(i,4) > -1
%             disp(i)
%             disp(tmap(i,4))
%             disp(cmap(tmap(i,4) + 1, :))
%             disp(length(tmap))
%             disp(length(coords))
            plot3([coords(i - 1, 1), coords(i, 1)], [coords(i - 1, 2), coords(i, 2)], [tmap(i, 4), tmap(i,4)], 'k-', 'LineWidth', 4, 'Color', cmap(tmap(i,4) + 1, :));
        end
        %pause(0.005);
    end
    
    hold off
    axis off
    
    % print -dpng -r300 image
end
