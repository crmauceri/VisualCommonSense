label_save_str = sprintf('%s/validation/multi_class_%s_balanced_random_processed.mat', root, search_description);
load(label_save_str);

codata = load(sprintf('%s/Flickr_cooccurrence/%s_pattern_analysis_owners.mat', root, search_description));
data = load(sprintf('%s/Flickr_concepts/%s_analysis.mat', root, search_description));

addpath('../analysis');
addpath('../prec_rec');
variable_def
metrics = [N_PMI, NUM_IMAGES, NUM_OWNERS];
metric_str = {'Normalized Pointwise Mutual Information', 'Number of Images', 'Number of Owners'};

ratio = codata.data(:, NUM_OWNERS) ./ codata.data(:, NUM_IMAGES);
cond_prob = reshape(codata.data(:, COND_PROB), codata.dataDims(1:2));
max_cond_prob =  max(cond_prob, cond_prob');
min_cond_prob =  min(cond_prob, cond_prob');

scores = load(sprintf('%s/validation/scores.mat', root));
score_mat = zeros(codata.dataDims(1:2));
score_mat(sub2ind(size(score_mat), scores.tag2_index, scores.tag1_index)) = scores.Score1;
score_mat(~isfinite(score_mat)) = max(score_mat(isfinite(score_mat)));
max_score = max(score_mat, score_mat');

% %Histogram
% mask = edge_label_mask;
% for ii=1:length(metrics)
%     figure;
%     metric = metrics(ii);
%     [N_incurve,edges] = histcounts(codata.data(mask(:), metric));
%     N = histcounts(codata.data(codata.data(:, metric)>0.1, metric), edges);
%     bar([N/max(N(:)); N_incurve/max(N_incurve(:))]');
%     set(gca, 'XTickLabelRotation', 90);
%     set(gca, 'XTick', 1:length(edges));
%     set(gca, 'XTickLabels', edges);
%     title(metric_str{ii});
%     legend({'All Examples', 'Hand-labeled Examples'});
%     xlabel('Value');
%     ylabel('Percent of Examples');
% end

for label = 2:3
    if label == 1
       mask = edge_label_mask > 0;
       label_mat = related;
       label_str = 'undirected edge existance';
    elseif label==2
       mask = any(edge_label(:, ~ismember(relationships, {'GeographicContainment', 'AtLocationGeographic'})), 2);
       label_mat = non_visual;
       label_str = 'visualness';
    else
       mask = edge_label_mask > 0;
       label_mat = any(edge_label(:,2:end), 2);
       label_str = 'directed edge existance';
    end

    plot_type = 'plotPR';
    h1 = figure; hold on;
    h2 = figure; hold on;
    h3 = figure; hold on;
    xlabel('False Positive Rate');
    ylabel('Threshold Value');
    for ii=1:length(metrics)
        metric = metrics(ii);
        figure(h1);
        [prec, tpr, fpr, thresh] = prec_rec(codata.data(mask(:), metric), label_mat(mask), plot_type, 1, 'holdFigure', 1);
        figure(h2);
        plot(fpr, thresh./max(codata.data(mask(:), metric)));
        figure(h3);
        plot(fpr, tpr);
    end

    figure(h1);
    [prec, tpr, fpr, thresh] = prec_rec(ratio(mask(:))', label_mat(mask)', plot_type, 1, 'holdFigure', 1);
    figure(h2);
    plot(fpr, thresh./max(ratio(mask(:))));
    figure(h3);
    plot(fpr, tpr);

    if label == 1
        figure(h1);
        [prec, tpr, fpr, thresh] = prec_rec(max_score(mask(:))', label_mat(mask)', plot_type, 1, 'holdFigure', 1);
        figure(h2);
        plot(fpr, thresh./max(max_score(mask(:))));
        figure(h3);
        plot(fpr, tpr);

        figure(h1);
        [prec, tpr, fpr, thresh] = prec_rec(max_cond_prob(mask(:))', label_mat(mask)', plot_type, 1, 'holdFigure', 1);
        figure(h2);
        plot(fpr, thresh);
        figure(h3);
        plot(fpr, tpr);

        figure(h1);
        [prec, tpr, fpr, thresh] = prec_rec(min_cond_prob(mask(:))', label_mat(mask)', plot_type, 1, 'holdFigure', 1);
        figure(h2);
        plot(fpr, thresh);
        figure(h3);
        plot(fpr, tpr);
    else
        figure(h1);
        [prec, tpr, fpr, thresh] = prec_rec(score_mat(mask(:))', label_mat(mask)', plot_type, 1, 'holdFigure', 1);
        figure(h2);
        plot(fpr, thresh./max(score_mat(mask(:))));
        figure(h3);
        plot(fpr, tpr);

        figure(h1);
        [prec, tpr, fpr, thresh] = prec_rec(cond_prob(mask(:))', label_mat(mask)', plot_type, 1, 'holdFigure', 1);
        figure(h2);
        plot(fpr, thresh);
        figure(h3);
        plot(fpr, tpr);
    end 

    if label == 1
        legend_str = {'Normalized PMI', 'Number of Images', 'Number of Unique Owners', 'Ratio Owners to Images', ...
            'Highest Classifier Score', 'Maximum Conditional Probability', 'Minimum Conditional Probatility'};
    else
        legend_str = {'Normalized PMI', 'Number of Images', 'Number of Unique Owners', 'Ratio Owners to Images', ...
            'Highest Classifier Score', 'Conditional Probability'};
    end
    
    figure(h1);
    %title(sprintf('Precision Recall curve predicting %s for animal dataset', label_str));
    xlabel('Recall');
    ylabel('Precision');
    legend(legend_str, 'Location', 'southeast');
    save_path = [root '/validation/' search_description '_' label_str '_' plot_type '.fig'];
    saveas(h1, save_path);
    save_path = ['C:/Users/mauceri2/Dropbox/UIUC/MastersThesis/figs/' search_description '_' label_str '_' plot_type '.png'];
    saveas(h1, save_path);

    figure(h2);
    legend(legend_str, 'Location', 'southeast');
    xlabel('False Positive Rate');
    ylabel('Threshold');
    save_path = [root '/validation/' search_description '_' label_str '_threshold.fig'];
    saveas(h2, save_path);
    save_path = ['C:/Users/mauceri2/Dropbox/UIUC/MastersThesis/figs/' search_description '_' label_str '_threshold.png'];
    saveas(h2, save_path);

    figure(h3);
    plot(0:1, 0:1, 'k');
    %title(sprintf('ROC predicting %s for animal dataset', label_str));
    legend(legend_str, 'Location', 'southeast');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    save_path = [root '/validation/' search_description '_' label_str '_plotROC.fig'];
    saveas(h3, save_path);
    save_path = ['C:/Users/mauceri2/Dropbox/UIUC/MastersThesis/figs/' search_description '_' label_str '_plotROC.png'];
    saveas(h3, save_path);
end

% figure; hold on;
% C = {'b','r','y', 'm', 'g', 'k'}; % Cell array of colors.
% for ii=1:length(metrics)
%     metric = metrics(ii);
%     [tpr,fpr,pmi_thresholds] = roc(related(edge_label_mask)',codata.data(edge_label_mask(:), metric)');
%     plot(fpr, pmi_thresholds ./ max(pmi_thresholds(:)), 'color', C{ii});
% end
% 
% [tpr,fpr,pmi_thresholds] = roc(related(non_inf_score)',score_mat(non_inf_score)');
% plot(fpr, pmi_thresholds ./ max(pmi_thresholds(:)), 'color', C{ii+1});
% 
% legend({'Conditional Probability', 'Normalized PMI', 'Number of Images', 'Number of Unique Owners', ...
%     'Highest Classifier Score'}, 'Location', 'northeast');
% title('Thresholds using metric to predict edge existance');
% xlabel('False Positive Rate');
% ylabel('Normalized Metric Threshold');
% 
% 
% figure; hold on;
% for ii=1:length(metrics)
%     metric = metrics(ii);
%     [tpr,fpr,pmi_thresholds] = roc(non_visual(edge_label_mask)',codata.data(edge_label_mask(:), metric)');
%     plot(fpr, pmi_thresholds ./ max(pmi_thresholds(:)), 'color', C{ii});
% end
% 
% [tpr,fpr,pmi_thresholds] = roc(non_visual(non_inf_score)',score_mat(non_inf_score)');
% plot(fpr, pmi_thresholds ./ max(pmi_thresholds(:)), 'color', C{ii+1});
% 
% legend({'Conditional Probability', 'Normalized PMI', 'Number of Images', 'Number of Unique Owners', ...
%     'Highest Classifier Score'}, 'Location', 'northeast');
% title('Thresholds using metric to predict visualness');
% xlabel('False Positive Rate');
% ylabel('Normalized Metric Threshold');

