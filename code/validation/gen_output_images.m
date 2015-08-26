function gen_output_images(root, search_description, query)
    addpath('analysis');
    addpath('transfer');

    concepts = load(sprintf('%s/concepts/%s_owner_per_concept_old.mat', root, search_description));
    concepts_old = concepts.concepts;

    concepts = load(sprintf('%s/concepts/%s_owner_per_concept.mat', root, search_description));
    concepts = concepts.concepts;

    label_save_str = sprintf('%s/validation/multi_class_%s_new.mat', root, search_description);

    num_concepts = length(concepts);
    if ~(exist( label_save_str , 'file'))
        edge_label_mask = zeros(num_concepts, num_concepts);
        edge_label = zeros(num_concepts, num_concepts);
        quality = zeros(num_concepts, num_concepts);
        non_visual = zeros(num_concepts, num_concepts);
        image_matrix = cell(500,1);
        image_index = zeros(2, 500);
        cur_edge = 1;
    else
        edge_labels = load(label_save_str);
        edge_label_mask = edge_labels.edge_label_mask;
        edge_label = edge_labels.edge_label;
        quality = edge_labels.quality;
        non_visual = edge_labels.non_visual;
        image_matrix = cell(500, 1);
        image_index = zeros(2, 500);
        cur_edge = 1;
    end

    [has_relations_mask, in_flickr, sorted_relations_str, ~, ~, ~, ~, ~, ~] ...
        = initialize_workspace_retrieve(root, search_description, true, 'glove.42B.300d', ...
        true, 100, 0, 0, false);

    has_relations_mask = reshape(has_relations_mask, num_concepts, num_concepts);
    in_flickr = reshape(in_flickr, num_concepts, num_concepts);

    all_edges = (has_relations_mask & in_flickr) | (~(has_relations_mask | has_relations_mask') & in_flickr);
    relationships = [sorted_relations_str; {'AtTime', 'Other'}'];

    image_axes = figure;
    set(image_axes, 'Position', [100 100 900 900]);
    
    num_images = 1;
    while num_images <= 500
        [image_matrix, image_index, num_images] = next_edge(root, query, all_edges, edge_label_mask, ...
            concepts_old, concepts, image_matrix, image_index, num_images, image_axes);
    end

    save(label_save_str, 'edge_label', 'edge_label_mask', 'quality', 'non_visual', 'image_matrix', 'image_index');

end

function [image_matrix, image_index, num_images] = next_edge(root, query, all_edges, edge_label_mask, ...
    concepts_old, concepts, image_matrix, image_index, num_images, image_axes)

    try
        [x, y] = find(all_edges & ~edge_label_mask);
        ind = randi(length(x));
        x = x(ind);
        y = y(ind);    
        image_index(:, num_images) = [x, y];

        tag1 = strrep(concepts{y}, ' ', '');
        tag2 = strrep(concepts{x}, ' ', '');

        [~, tag1Ind] = ismember(tag1, concepts_old);
        [~, tag2Ind] = ismember(tag2, concepts_old);

        %Subtract 1 because of 1 based indexing in Matlab
        sortedInd = sort([tag1Ind, tag2Ind]-1, 'descend');

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
                image_id = image_ids{ii}{1};
                allI(:,:, :, ii) = insertText(imresize(I, [100 100]), [10, 10], image_id(2:end-1));
            end
            toc;

            montage(allI);
            F = getframe(gcf);
            [X, Map] = frame2im(F);
            image_matrix{num_images} = X;
            num_images = num_images + 1;
        end
    catch 
        display(sprintf('Error loading next edge: %s, %s', concepts{y}, concepts{x}));
    end

end