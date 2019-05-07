function loops = detect_loops(loops_file, prev, cons_loops, inliers, nAgents)

    curr_loops = loops_file;
    curr_loops_size = size(curr_loops);
    nimages = curr_loops_size(1);
    loops = zeros(nimages, 4);
    
    % Processing each image to generate the corresponding response
    consecutive_loops = zeros(1, nAgents);
    for i=1:nimages
        currAgent = curr_loops(i, 4);
        overlap = curr_loops(i, 3) == 1;
        if curr_loops(i, 1) == -1 && curr_loops(i, 2) == -1 && curr_loops(i, 3) == -1 && curr_loops(i, 4) == -1 
            % Discarding non porcessed images
            loops(i, 1) = i - 1;
            loops(i, 2) = 2;
            loops(i, 3) = 0;
            loops(i, 4) = 0;
        elseif curr_loops(i, 1) == 0 && curr_loops(i, 2) == 0
            % Assesing if there are no islands
            loops(i, 1) = i - 1;
            loops(i, 2) = 3;
            loops(i, 3) = 0;
            loops(i, 4) = 0;
        else
            if consecutive_loops > cons_loops && overlap
                % Assuming loops in extreme conditions
                loops(i, 1) = i - 1;
                loops(i, 2) = 0;
                loops(i, 3) = curr_loops(i, 3);
                loops(i, 4) = 0;                
                consecutive_loops = consecutive_loops + 1;
            else
                if curr_loops(i, 5) > inliers
                    % Correct loop due to inliers
                    loops(i, 1) = i - 1;
                    loops(i, 2) = 0;
                    loops(i, 3) = curr_loops(i, 3);
                    loops(i, 4) = curr_loops(i, 5);
                    consecutive_loops = consecutive_loops + 1;
                else
                    % Incorrect loop due to there are not enough inliers
                    loops(i, 1) = i - 1;
                    loops(i, 2) = 4;
                    loops(i, 3) = curr_loops(i, 3);
                    loops(i, 4) = curr_loops(i, 5);
                    consecutive_loops = 0;
                end
            end            
        end
    end
end