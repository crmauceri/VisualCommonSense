function [ accuracy, precision, recall ] = accuracy_precision_recall( ground_truth, predicted )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

accuracy = sum(predicted==ground_truth)/ length(ground_truth);
precision = sum(predicted & ground_truth)/ sum(predicted);
recall = sum(predicted & ground_truth) / sum(ground_truth);

end

