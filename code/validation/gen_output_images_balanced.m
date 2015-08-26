function gen_output_images_balanced(dataroot, search_description, query)
    addpath('../analysis');
    addpath('../transfer');
    variable_def;

    concepts = load(sprintf('%s/concepts/%s_owner_per_concept.mat', dataroot, search_description));
    concepts = cellstr(concepts.concepts);
    num_concepts = length(concepts);

    codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', dataroot, search_description));
    
    if ~exist(sprintf('%s/validation/', dataroot))
        mkdir(sprintf('%s/validation/', dataroot));
    end
    label_save_str = sprintf('%s/validation/multi_class_%s_balanced_random2.mat', dataroot, search_description);
    
    [~, in_flickr, sorted_relations_str, ~, ~, ~, ~, ~, ~] ...
        = initialize_workspace_retrieve(dataroot, search_description, true, 'glove.42B.300d', ...
        true, 100, 50, 0, false);
    
    in_flickr = reshape(in_flickr, num_concepts, num_concepts);
    all_edges = in_flickr;
    
    label_load = sprintf('%s/validation/multi_class_%s_balanced_random_processed.mat', dataroot, search_description);
    if exist(label_load)
        processed = load(label_load);
        all_edges(processed.edge_label_mask > 0) = 0;
    end
    
    num_images_per_metric = 50;
    relationships = [{'None'}; sorted_relations_str; {'AtTime', 'LooksLike', 'Other'}'];
    num_relationships = length(relationships);
    if ~(exist( label_save_str , 'file'))
        edge_label_mask = sparse(num_concepts, num_concepts);
        edge_label = sparse(num_concepts*num_concepts, num_relationships);
        related = sparse(num_concepts, num_concepts);
        non_visual = sparse(num_concepts, num_concepts);
        final_image_matrix = cell(num_images_per_metric*5,1);
        final_image_index = zeros(2, num_images_per_metric*5);
        cur_edge = 1;
        dim = [num_concepts, num_concepts];
    else
        edge_labels = load(label_save_str);
        edge_label_mask = edge_labels.edge_label_mask;
        edge_label = edge_labels.edge_label;
        related = edge_labels.related;
        non_visual = edge_labels.non_visual;
        dim = edge_labels.dim;
        final_image_matrix = edge_labels.final_image_matrix;
        final_image_index = edge_labels.final_image_index;
        cur_edge = 1;
    end

    image_axes = figure;
    set(image_axes, 'Position', [100 100 900 900]);
    
    try
        tested_edges = [];
        metrics = [COND_PROB, N_PMI, NUM_IMAGES, NUM_OWNERS];
        for ii = 1:4
            metric = metrics(ii);
            data = full(codata.data(all_edges(:), metric));
            edge_ind = find(all_edges(:));
            [image_matrix, image_index, tested_edges] = select_edge(data, edge_ind, ...
                tested_edges, dataroot, query, concepts, dim, num_images_per_metric);
            final_image_matrix((ii-1)*num_images_per_metric+1:ii*num_images_per_metric) = image_matrix;
            final_image_index(:, (ii-1)*num_images_per_metric+1:ii*num_images_per_metric) = image_index;
        end

        load(sprintf('%s/validation/scores.mat', dataroot));
        edge_ind = sub2ind(dim, tag2_index, tag1_index);
        [image_matrix, image_index, tested_edges] = select_edge(Score1(2:end), edge_ind, ...
            tested_edges, dataroot, query, concepts, dim, num_images_per_metric);
        final_image_matrix(num_images_per_metric*4+1:num_images_per_metric*5) = image_matrix;
        final_image_index(:, num_images_per_metric*4+1:num_images_per_metric*5) = image_index;
        save(label_save_str, 'edge_label', 'edge_label_mask', 'related', 'non_visual', 'final_image_matrix', 'final_image_index', 'dim', 'relationships');
    catch
        display('Ended on error')
        save(label_save_str, '-v7.3');
    end        
end

function [image_matrix, image_index, tested_edges] = select_edge(data, edge_ind, ...
    tested_edges, root, query, concepts, dim, num_images_per_metric)
    
    num_bins = 50;
    [~, centers] = hist(data, num_bins);
    [~, bin_ind] = histc(data, centers);
    num_images = 1;
    min_count = 1;
    image_matrix = cell(num_images_per_metric, 1);
    image_index = zeros(2, num_images_per_metric);
    
    potential_edges = cell(num_bins, 1);
    for cur_bin = 1:num_bins
        cur_edge = edge_ind(bin_ind == cur_bin);
        untested = ~ismember(cur_edge, tested_edges);
        potential_edges{cur_bin} = cur_edge(untested);
    end
    
    while num_images <= num_images_per_metric
        for cur_bin = 1:num_bins
            if num_images <= num_images_per_metric & length(potential_edges{cur_bin})>0
                cur_edge = potential_edges{cur_bin}(randi(length(potential_edges{cur_bin})));
                if ~ismember(cur_edge, tested_edges)
                   tested_edges = [tested_edges, cur_edge];
                   [x, y] = ind2sub(dim, cur_edge);
                   [image_matrix, image_index, num_images] = next_edge(root, query, x, y, ...
                        concepts, image_matrix, image_index, num_images);
                end
            else
                break
            end
        end
    end
end


function [image_matrix, image_index, num_images] = next_edge(root, query, x, y, ...
    concepts, image_matrix, image_index, num_images)

    try   
        image_index(:, num_images) = [x, y];

        tag1 = strrep(concepts{y}, ' ', '');
        tag2 = strrep(concepts{x}, ' ', '');

        %Subtract 1 because of 1 based indexing in Matlab
        sortedInd = sort([y, x]-1, 'descend');

        display('Load index')
        tic;
        pair_index = load_image_pair_index(root, query, sortedInd(1));
        toc;

        command = sprintf('images = pair_index.s%d;', sortedInd(2));
        eval(command);

        max_ind = min(25, length(images));
        
        if max_ind == 25
            allI = zeros(100, 100, 3, max_ind);
            curimages = images(1:max_ind);
            image_ids = cellfun(@(x) regexp(x, '/\w*_', 'match'), curimages, 'UniformOutput', false); 

            display('Load images');
            tic;
            for ii=1:max_ind   
                I = im2double(imread(curimages{ii}));
                if length(size(I))<3
                    I = repmat(I, 1, 1, 3);
                end
                allI(:,:, :, ii) = imresize(I, [100 100]);
%                 image_id = image_ids{ii}{1};
%                 allI(:,:, :, ii) = insertText(imresize(I, [100 100]), [10, 10], image_id(2:end-1));
            end
            toc;

            montage(allI);
            hold on;
            for ii = 1:max_ind
                image_id = image_ids{ii}{1};
                [x, y] = ind2sub([5, 5], ii);
                rectangle('Position', [(x-1)*100 + 10, (y-1)*100+5, 80, 10], 'FaceColor', 'w');
                text((x-1)*100 + 10, (y-1)*100+10, 1, image_id(2:end-1));
            end
            F = getframe(gcf);
            [X, Map] = frame2im(F);
            clf;
            image_matrix{num_images} = X;
            num_images = num_images + 1;
        end
    catch 
        display(sprintf('Error loading next edge: %s, %s', concepts{y}, concepts{x}));
    end

end