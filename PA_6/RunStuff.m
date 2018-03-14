[G, F] = ConstructToyNetwork(1.0, 0.2);
randi('seed',1);
for i = 1:3
    [M all_samples] = MCMCInference(G, F, [], 'MHSwendsenWang1', 1, 12000, 1, repmat(i, 1, 16));
    samples_list{i} = all_samples; 
end
VisualizeMCMCMarginals(samples_list, 1:length(G.names), G.card, F, 1500, 'MHSwendsenWang1');