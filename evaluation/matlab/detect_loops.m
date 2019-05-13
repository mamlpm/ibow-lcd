function loops = detect_loops(loops_file, cons_loops, inliers, nAgents)

    curr_loops = loops_file;
    curr_loops_size = size(curr_loops);
    nimages = curr_loops_size(1);
    loops = zeros(nimages, 4);
    
    % Processing each image to generate the corresponding response
    consecutive_loops = zeros(1, nAgents);
    for i=1:nimages
        currAgent = curr_loops(i, 4) + 1;
        overlap = curr_loops(i, 3) == 1;
        if curr_loops(i, 1) == -1 && curr_loops(i, 2) == -1 && curr_loops(i, 3) == -1 && curr_loops(i, 4) == -1 
            % Discarding non processed images
            loops(i, 1) = max(i - 1, curr_loops(i,1));
            loops(i, 2) = 2;
            loops(i, 3) = 0;
            loops(i, 4) = 0;
%         elseif curr_loops(i, 1) == 0 && curr_loops(i, 2) == 0
%             % Assesing if there are no islands
%             loops(i, 1) = i - 1;
%             loops(i, 2) = 3;
%             loops(i, 3) = 0;
%             loops(i, 4) = 0;
        else
            if consecutive_loops(currAgent) > cons_loops && overlap
                % Assuming loops in extreme conditions
                loops(i, 1) = max(i - 1, curr_loops(i,1));
                loops(i, 2) = 0;
%                 loops(i, 3) = curr_loops(i, 3);
                loops(i, 3) = min(curr_loops(i, 1), i - 1);
                loops(i, 4) = 0;                
                consecutive_loops(currAgent) = consecutive_loops(currAgent) + 1;
            else
                if curr_loops(i, 2) > inliers
                    % Correct loop due to inliers
                    loops(i, 1) = max(i - 1, curr_loops(i,1));
                    loops(i, 2) = 0;
                    loops(i, 3) = min(curr_loops(i, 1), i - 1);
                    loops(i, 4) = curr_loops(i, 2);
                    consecutive_loops(currAgent) = consecutive_loops(currAgent) + 1;
                else
                    % Incorrect loop due to there are not enough inliers
                    loops(i, 1) = max(i - 1, curr_loops(i,1));
                    loops(i, 2) = 4;
                    loops(i, 3) = min(curr_loops(i, 1), i - 1);
                    loops(i, 4) = curr_loops(i, 2);
                    consecutive_loops(currAgent) = 0;
                end
            end            
        end
    end
end